//go:build race

package demobot

// raceBuild reports a -race build. Sweeps that exercise no concurrency trim
// their redundant axes under it: the race detector checks nothing there and
// multiplies their cost.
const raceBuild = true
