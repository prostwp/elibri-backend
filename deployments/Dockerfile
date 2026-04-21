FROM golang:1.25-alpine AS builder
RUN apk add --no-cache git
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /server ./cmd/server

FROM alpine:3.19
RUN apk add --no-cache ca-certificates tzdata
# Runtime must find migrations via the relative path baked into postgres.go
# ("internal/store/migrations/*.sql"). Keep WORKDIR = /app and copy the SQL
# tree so migration runner sees them on container start.
WORKDIR /app
COPY --from=builder /server /app/server
COPY --from=builder /app/internal/store/migrations /app/internal/store/migrations
# ML models are ~1.9GB and are not shipped inside the image. Mount them at
# /app/ml-training/models via volume, or bake a smaller subset via a
# separate build stage if needed.
EXPOSE 8080
CMD ["/app/server"]
