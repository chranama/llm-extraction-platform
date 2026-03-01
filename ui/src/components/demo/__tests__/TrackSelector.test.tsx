import React from "react";
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { TrackSelector } from "../TrackSelector";

describe("TrackSelector", () => {
  it("switches tracks and triggers refresh handlers", () => {
    const onChangeTrack = vi.fn();
    const onRefreshNow = vi.fn();
    const onToggleAutoRefresh = vi.fn();
    const onChangeRefreshEverySeconds = vi.fn();

    render(
      <TrackSelector
        track="generate_clamp"
        onChangeTrack={onChangeTrack}
        apiBase="/api"
        hasApiKey={true}
        autoRefresh={false}
        onToggleAutoRefresh={onToggleAutoRefresh}
        refreshEverySeconds={5}
        onChangeRefreshEverySeconds={onChangeRefreshEverySeconds}
        onRefreshNow={onRefreshNow}
        isRefreshing={false}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: /Extract gating/i }));
    expect(onChangeTrack).toHaveBeenCalledWith("extract_gating");

    fireEvent.click(screen.getByRole("button", { name: /Refresh now/i }));
    expect(onRefreshNow).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("checkbox", { name: /Auto-refresh/i }));
    expect(onToggleAutoRefresh).toHaveBeenCalledTimes(1);

    fireEvent.change(screen.getByRole("spinbutton"), { target: { value: "12" } });
    expect(onChangeRefreshEverySeconds).toHaveBeenCalledWith(12);
  });

  it("disables refresh button when refreshing", () => {
    render(
      <TrackSelector
        track="generate_clamp"
        onChangeTrack={vi.fn()}
        apiBase="/api"
        hasApiKey={false}
        autoRefresh={true}
        onToggleAutoRefresh={vi.fn()}
        refreshEverySeconds={5}
        onChangeRefreshEverySeconds={vi.fn()}
        onRefreshNow={vi.fn()}
        isRefreshing={true}
      />
    );

    expect(screen.getByRole("button", { name: /Refreshing/i })).toBeDisabled();
    expect(screen.getByText(/API key: missing/i)).toBeInTheDocument();
  });
});

