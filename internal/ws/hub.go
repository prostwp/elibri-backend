package ws

import (
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin:     func(r *http.Request) bool { return true },
}

// Message sent over WebSocket
type Message struct {
	Type   string      `json:"type"`
	Topic  string      `json:"topic,omitempty"`
	Data   interface{} `json:"data,omitempty"`
	Topics []string    `json:"topics,omitempty"`
}

// Client represents a single WS connection
type Client struct {
	hub    *Hub
	conn   *websocket.Conn
	send   chan []byte
	topics map[string]bool
	mu     sync.RWMutex
}

// Hub manages all WS clients and broadcasting
type Hub struct {
	clients    map[*Client]bool
	topics     map[string]map[*Client]bool
	broadcast  chan BroadcastMsg
	register   chan *Client
	unregister chan *Client
	mu         sync.RWMutex
}

type BroadcastMsg struct {
	Topic string
	Data  []byte
}

func NewHub() *Hub {
	return &Hub{
		clients:    make(map[*Client]bool),
		topics:     make(map[string]map[*Client]bool),
		broadcast:  make(chan BroadcastMsg, 256),
		register:   make(chan *Client),
		unregister: make(chan *Client),
	}
}

func (h *Hub) Run() {
	for {
		select {
		case client := <-h.register:
			h.mu.Lock()
			h.clients[client] = true
			h.mu.Unlock()
			log.Printf("WS client connected (%d total)", len(h.clients))

		case client := <-h.unregister:
			h.mu.Lock()
			if _, ok := h.clients[client]; ok {
				delete(h.clients, client)
				close(client.send)
				// Remove from all topics
				for topic, subs := range h.topics {
					delete(subs, client)
					if len(subs) == 0 {
						delete(h.topics, topic)
					}
				}
			}
			h.mu.Unlock()
			log.Printf("WS client disconnected (%d total)", len(h.clients))

		case msg := <-h.broadcast:
			h.mu.RLock()
			if subs, ok := h.topics[msg.Topic]; ok {
				for client := range subs {
					select {
					case client.send <- msg.Data:
					default:
						// Client buffer full, skip
					}
				}
			}
			h.mu.RUnlock()
		}
	}
}

// Publish sends a message to all clients subscribed to a topic
func (h *Hub) Publish(topic string, data interface{}) {
	msg := Message{Type: "data", Topic: topic, Data: data}
	bytes, err := json.Marshal(msg)
	if err != nil {
		return
	}
	h.broadcast <- BroadcastMsg{Topic: topic, Data: bytes}
}

// HandleWS upgrades HTTP to WebSocket
func (h *Hub) HandleWS(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WS upgrade error: %v", err)
		return
	}

	client := &Client{
		hub:    h,
		conn:   conn,
		send:   make(chan []byte, 256),
		topics: make(map[string]bool),
	}

	h.register <- client

	go client.writePump()
	go client.readPump()
}

func (c *Client) readPump() {
	defer func() {
		c.hub.unregister <- c
		c.conn.Close()
	}()

	c.conn.SetReadLimit(4096)
	c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	c.conn.SetPongHandler(func(string) error {
		c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	for {
		_, message, err := c.conn.ReadMessage()
		if err != nil {
			break
		}

		var msg Message
		if err := json.Unmarshal(message, &msg); err != nil {
			continue
		}

		switch msg.Type {
		case "subscribe":
			c.hub.mu.Lock()
			for _, topic := range msg.Topics {
				if _, ok := c.hub.topics[topic]; !ok {
					c.hub.topics[topic] = make(map[*Client]bool)
				}
				c.hub.topics[topic][c] = true
				c.mu.Lock()
				c.topics[topic] = true
				c.mu.Unlock()
			}
			c.hub.mu.Unlock()

		case "unsubscribe":
			c.hub.mu.Lock()
			for _, topic := range msg.Topics {
				if subs, ok := c.hub.topics[topic]; ok {
					delete(subs, c)
				}
				c.mu.Lock()
				delete(c.topics, topic)
				c.mu.Unlock()
			}
			c.hub.mu.Unlock()

		case "ping":
			resp, _ := json.Marshal(Message{Type: "pong"})
			c.send <- resp
		}
	}
}

func (c *Client) writePump() {
	ticker := time.NewTicker(54 * time.Second)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()

	for {
		select {
		case message, ok := <-c.send:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if !ok {
				c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}
			c.conn.WriteMessage(websocket.TextMessage, message)

		case <-ticker.C:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}
