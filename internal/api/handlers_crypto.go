package api

import (
	"net/http"

	"github.com/prostwp/elibri-backend/internal/crypto"
)

func handleCryptoScanReal(w http.ResponseWriter, r *http.Request) {
	results, err := crypto.RunScan(crypto.DefaultScanConfig)
	if err != nil {
		http.Error(w, `{"error":"`+err.Error()+`"}`, http.StatusInternalServerError)
		return
	}

	writeJSON(w, map[string]interface{}{
		"results":    results,
		"count":      len(results),
		"scan_type":  "dip_scanner",
	})
}

func handleCryptoListingsReal(w http.ResponseWriter, r *http.Request) {
	listings, err := crypto.CheckNewListings()
	if err != nil {
		http.Error(w, `{"error":"`+err.Error()+`"}`, http.StatusInternalServerError)
		return
	}

	writeJSON(w, map[string]interface{}{
		"new_listings": listings,
		"known_count":  crypto.GetKnownCount(),
	})
}
