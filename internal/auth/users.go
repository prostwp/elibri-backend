package auth

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type User struct {
	ID           string    `json:"id"`
	Email        string    `json:"email"`
	PasswordHash string    `json:"-"`
	DisplayName  string    `json:"display_name"`
	Role         string    `json:"role"`
	CreatedAt    time.Time `json:"created_at"`
	UpdatedAt    time.Time `json:"updated_at"`
}

var ErrUserExists = errors.New("user already exists")
var ErrUserNotFound = errors.New("user not found")
var ErrInvalidCredentials = errors.New("invalid credentials")

// CreateUser inserts a new user. Returns ErrUserExists on duplicate email.
func CreateUser(ctx context.Context, pool *pgxpool.Pool, email, password, displayName, role string) (*User, error) {
	email = strings.ToLower(strings.TrimSpace(email))
	if role == "" {
		role = "user"
	}

	hash, err := HashPassword(password)
	if err != nil {
		return nil, err
	}

	var u User
	err = pool.QueryRow(ctx, `
		INSERT INTO users (email, password_hash, display_name, role)
		VALUES ($1, $2, $3, $4)
		RETURNING id, email, password_hash, COALESCE(display_name, ''), role, created_at, updated_at
	`, email, hash, displayName, role).Scan(
		&u.ID, &u.Email, &u.PasswordHash, &u.DisplayName, &u.Role, &u.CreatedAt, &u.UpdatedAt,
	)
	if err != nil {
		if strings.Contains(err.Error(), "duplicate key") || strings.Contains(err.Error(), "users_email_key") {
			return nil, ErrUserExists
		}
		return nil, err
	}
	return &u, nil
}

// GetUserByEmail returns user by email. ErrUserNotFound if missing.
func GetUserByEmail(ctx context.Context, pool *pgxpool.Pool, email string) (*User, error) {
	email = strings.ToLower(strings.TrimSpace(email))
	var u User
	err := pool.QueryRow(ctx, `
		SELECT id, email, password_hash, COALESCE(display_name, ''), role, created_at, updated_at
		FROM users WHERE email = $1
	`, email).Scan(&u.ID, &u.Email, &u.PasswordHash, &u.DisplayName, &u.Role, &u.CreatedAt, &u.UpdatedAt)
	if err == pgx.ErrNoRows {
		return nil, ErrUserNotFound
	}
	if err != nil {
		return nil, err
	}
	return &u, nil
}

func GetUserByID(ctx context.Context, pool *pgxpool.Pool, id string) (*User, error) {
	var u User
	err := pool.QueryRow(ctx, `
		SELECT id, email, password_hash, COALESCE(display_name, ''), role, created_at, updated_at
		FROM users WHERE id = $1
	`, id).Scan(&u.ID, &u.Email, &u.PasswordHash, &u.DisplayName, &u.Role, &u.CreatedAt, &u.UpdatedAt)
	if err == pgx.ErrNoRows {
		return nil, ErrUserNotFound
	}
	if err != nil {
		return nil, err
	}
	return &u, nil
}

func ListUsers(ctx context.Context, pool *pgxpool.Pool) ([]User, error) {
	rows, err := pool.Query(ctx, `
		SELECT id, email, '' AS password_hash, COALESCE(display_name, ''), role, created_at, updated_at
		FROM users ORDER BY created_at DESC
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	users := []User{}
	for rows.Next() {
		var u User
		if err := rows.Scan(&u.ID, &u.Email, &u.PasswordHash, &u.DisplayName, &u.Role, &u.CreatedAt, &u.UpdatedAt); err != nil {
			return nil, err
		}
		users = append(users, u)
	}
	return users, rows.Err()
}

// Authenticate returns user if credentials match, else ErrInvalidCredentials.
func Authenticate(ctx context.Context, pool *pgxpool.Pool, email, password string) (*User, error) {
	u, err := GetUserByEmail(ctx, pool, email)
	if err == ErrUserNotFound {
		return nil, ErrInvalidCredentials
	}
	if err != nil {
		return nil, err
	}
	if !VerifyPassword(u.PasswordHash, password) {
		return nil, ErrInvalidCredentials
	}
	return u, nil
}

// ResetUserPassword updates password_hash. Used by admin.
func ResetUserPassword(ctx context.Context, pool *pgxpool.Pool, userID, newPassword string) error {
	hash, err := HashPassword(newPassword)
	if err != nil {
		return err
	}
	tag, err := pool.Exec(ctx, `
		UPDATE users SET password_hash = $1, updated_at = NOW() WHERE id = $2
	`, hash, userID)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return ErrUserNotFound
	}
	return nil
}

// PromoteToAdminIfAllowed upgrades role to admin when email matches allowlist.
func PromoteToAdminIfAllowed(ctx context.Context, pool *pgxpool.Pool, email string, adminEmails []string) {
	email = strings.ToLower(strings.TrimSpace(email))
	for _, allowed := range adminEmails {
		if strings.ToLower(strings.TrimSpace(allowed)) == email {
			_, _ = pool.Exec(ctx, `UPDATE users SET role = 'admin', updated_at = NOW() WHERE email = $1`, email)
			return
		}
	}
}
