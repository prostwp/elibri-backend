package demobot

// radar.go — the Risk Radar agent's only I/O: reading the file an external
// Python job writes once a day (00:00 UTC) with a BTC daily-volatility-SIZE
// forecast, produced by a gradient-boosted model trained and validated
// offline. This agent never computes a forecast itself — it reads the file,
// checks it is for the caller's own UTC day, and hands it to radar_text.go
// for wording. See docs/demobot-http.md ("Risk Radar") for the file schema.

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"
)

// DefaultRadarFile is where DEMOBOT_RADAR_FILE points when unset — the path
// the Python job writes on the production host.
const DefaultRadarFile = "/root/flagship/results/radar.json"

// radarFilePath resolves DEMOBOT_RADAR_FILE, falling back to DefaultRadarFile.
func radarFilePath() string {
	if p := strings.TrimSpace(os.Getenv("DEMOBOT_RADAR_FILE")); p != "" {
		return p
	}
	return DefaultRadarFile
}

// radarValidation is the file's "validation" object: the model's backtest
// against 154 (as of this file's generation) unseen days.
type radarValidation struct {
	Days              int     `json:"days"`
	From              string  `json:"from"`
	To                string  `json:"to"`
	GainVsHAR         float64 `json:"gain_vs_har"`
	PVsHAR            float64 `json:"p_vs_har"`
	GainVsATR14       float64 `json:"gain_vs_atr14"`
	GainVsHARDow      float64 `json:"gain_vs_har_dow"`
	PVsHARDow         float64 `json:"p_vs_har_dow"`
	PreRegisteredPass bool    `json:"pre_registered_pass"`
}

// radarForward is the file's "forward" object: the live (post-registration)
// forward test, still accumulating days — GainVsHARDow is nil until Days > 0.
type radarForward struct {
	Days         int      `json:"days"`
	GainVsHARDow *float64 `json:"gain_vs_har_dow"`
}

// radarFile is the exact shape the external Python job writes. Every number
// the card prints comes from here — nothing in radar_text.go is a stand-in
// constant for a result the model produced.
type radarFile struct {
	Asset          string          `json:"asset"`
	ForecastDay    string          `json:"forecast_day"` // YYYY-MM-DD, UTC
	GeneratedAt    string          `json:"generated_at"` // RFC3339
	VolPct         float64         `json:"vol_pct"`
	YesterdayPct   float64         `json:"yesterday_pct"`
	Percentile180d float64         `json:"percentile_180d"`
	Level          string          `json:"level"`
	Validation     radarValidation `json:"validation"`
	Forward        radarForward    `json:"forward"`
}

// radarFileCache re-reads the file only when its mtime (or path) changed —
// the file is tiny and the job writes it once a day, so a GET every few
// seconds must not re-stat-then-parse on every request for nothing new.
type radarFileCache struct {
	mu      sync.Mutex
	loaded  bool
	path    string
	modTime time.Time
	file    radarFile
	err     error
}

func newRadarFileCache() *radarFileCache { return &radarFileCache{} }

// read returns the parsed file, re-reading only on a path or mtime change.
// A stat/read/parse failure is returned as-is and never cached across a
// filesystem change: the next call always re-tries.
func (c *radarFileCache) read(path string) (radarFile, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	info, statErr := os.Stat(path)
	if statErr != nil {
		c.loaded = false
		return radarFile{}, statErr
	}
	if c.loaded && c.path == path && info.ModTime().Equal(c.modTime) {
		return c.file, c.err
	}

	raw, readErr := os.ReadFile(path)
	if readErr != nil {
		c.loaded = false
		return radarFile{}, readErr
	}
	var f radarFile
	parseErr := json.Unmarshal(raw, &f)
	c.path, c.modTime, c.file, c.err, c.loaded = path, info.ModTime(), f, parseErr, true
	if parseErr != nil {
		return radarFile{}, fmt.Errorf("radar file %s: %w", path, parseErr)
	}
	return f, nil
}

// radarAssetOK validates the optional ?asset= against Risk Radar's coverage:
// BTC only, whatever alias resolves to it — everything else is the same
// honest "unknown asset" 400 every other agent's registry gives.
func radarAssetOK(arg string) error {
	if strings.TrimSpace(arg) == "" {
		return nil
	}
	key, err := resolveAssetKey(arg)
	if err != nil {
		return err
	}
	if key != "btc" {
		return fmt.Errorf("unknown asset %q — Risk Radar covers BTC only", arg)
	}
	return nil
}

// RadarCard reads the daily forecast file and formats it, or degrades
// honestly: no file / unparseable JSON reads like any other dead source
// (radarOfflineCard, a 503); a file whose forecast is not for the caller's
// own UTC day reads "no forecast yet" (statusNoData, a 200 with ok:false) —
// never yesterday's number relabeled as today's.
//
// The Volatility fusion line (radar_text.go) is built from a.VolCard for
// BTC — the exact card the Volatility Agent itself serves, sharing its
// kline cache — so the two agents can never say something different about
// the same candles.
func (a *Agents) RadarCard(ctx context.Context) Card {
	f, err := a.radar.read(radarFilePath())
	if err != nil {
		return radarOfflineCard()
	}
	now := a.clock()
	if f.ForecastDay != now.Format("2006-01-02") {
		return radarNoForecastCard(f.ForecastDay, now)
	}
	vol := a.VolCard(ctx, btcSpec)
	return radarCardFrom(f, radarVolatilityLine(vol), now)
}
