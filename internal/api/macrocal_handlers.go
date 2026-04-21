package api

import (
	"net/http"
	"strconv"
	"time"

	"github.com/prostwp/elibri-backend/internal/macrocal"
)

// macroCfg is the blackout config injected from main.go so the endpoint
// can use the same impact filter and window as the scenario runner.
var macroCfg macrocal.Config

// SetMacroConfig is called once from main.go.
func SetMacroConfig(cfg macrocal.Config) { macroCfg = cfg }

// GET /api/v1/macrocal?hours=24
// Returns upcoming macro events the scenario runner will respect for
// blackouts. Also reports the current blackout state so the Toolbar chip
// can show "🛑 FOMC in 25m" without a second roundtrip.
func handleMacroCalendar(w http.ResponseWriter, r *http.Request) {
	hours := 24
	if s := r.URL.Query().Get("hours"); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n > 0 && n <= 168 {
			hours = n
		}
	}

	events := macrocal.UpcomingEvents(hours, macroCfg)
	gate := macrocal.IsBlackout(time.Now().UTC(), macroCfg)

	resp := map[string]any{
		"events":           events,
		"blackout_active":  gate.Blocked,
		"blackout_event":   gate.Event,
		"blackout_minutes": gate.Minutes,
		"config": map[string]any{
			"enabled":        macroCfg.Enabled,
			"before_minutes": int(macroCfg.BlackoutBefore.Minutes()),
			"after_minutes":  int(macroCfg.BlackoutAfter.Minutes()),
			"min_impact":     macroCfg.MinImpact,
		},
		"last_fetch": macrocal.LastFetch(),
	}
	writeJSON(w, resp)
}
