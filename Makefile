.PHONY: build run test docker-up docker-down

build:
	go build -o bin/server ./cmd/server

run:
	go run ./cmd/server

test:
	go test ./...

docker-up:
	cd deployments && docker compose up -d --build

docker-down:
	cd deployments && docker compose down
