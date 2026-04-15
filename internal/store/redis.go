package store

import (
	"context"
	"encoding/json"
	"log"
	"time"

	"github.com/redis/go-redis/v9"
)

var Redis *redis.Client

func InitRedis(redisURL string) {
	opt, err := redis.ParseURL(redisURL)
	if err != nil {
		log.Printf("WARNING: Redis URL invalid: %v", err)
		return
	}

	Redis = redis.NewClient(opt)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	if err := Redis.Ping(ctx).Err(); err != nil {
		log.Printf("WARNING: Redis not available: %v", err)
		Redis = nil
		return
	}

	log.Println("Redis connected")
}

// CacheSet stores a value with TTL
func CacheSet(key string, value interface{}, ttl time.Duration) {
	if Redis == nil {
		return
	}
	data, err := json.Marshal(value)
	if err != nil {
		return
	}
	Redis.Set(context.Background(), key, data, ttl)
}

// CacheGet retrieves a cached value
func CacheGet(key string, dest interface{}) bool {
	if Redis == nil {
		return false
	}
	data, err := Redis.Get(context.Background(), key).Bytes()
	if err != nil {
		return false
	}
	return json.Unmarshal(data, dest) == nil
}

func CloseRedis() {
	if Redis != nil {
		Redis.Close()
	}
}
