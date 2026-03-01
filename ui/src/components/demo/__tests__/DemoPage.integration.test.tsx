import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";

vi.mock("../../../lib/api", async () => {
  const mod = await vi.importActual<typeof import("../../../lib/api")>("../../../lib/api");
  return {
    ...mod,
    listModels: vi.fn(),
    adminGetPolicy: vi.fn(),
    adminListLogs: vi.fn(),
    adminWriteGenerateSlo: vi.fn(),
    adminReloadPolicy: vi.fn(),
    adminReloadRuntime: vi.fn(),
    getApiBaseUrl: vi.fn(),
  };
});

vi.mock("../TrackSelector", () => ({
  TrackSelector: (props: any) => (
    <div data-testid="track-selector">
      <div>track:{props.track}</div>
      <button onClick={() => props.onChangeTrack("extract_gating")}>switch-track</button>
      <button onClick={props.onRefreshNow}>refresh-now</button>
      <button onClick={props.onToggleAutoRefresh}>toggle-auto-refresh</button>
      <button onClick={() => props.onChangeRefreshEverySeconds(2)}>set-refresh-2s</button>
    </div>
  ),
}));

vi.mock("../StatePanel", () => ({
  StatePanel: (props: any) => (
    <div data-testid="state-panel">
      <div>model:{props.models?.default_model ?? "none"}</div>
      <div>effective:{String(props.effectiveExtractEnabled)}</div>
      <button onClick={props.onRefreshModels}>refresh-models</button>
      <button onClick={props.onRefreshPolicy}>refresh-policy</button>
    </div>
  ),
}));

vi.mock("../ActionsPanel", () => ({
  ActionsPanel: (props: any) => (
    <div data-testid="actions-panel">
      <div>disabled:{String(props.disabled)}</div>
      <button onClick={props.onWriteGenerateSlo}>write-slo</button>
      <button onClick={props.onReloadPolicy}>reload-policy</button>
      <button onClick={props.onReloadRuntime}>reload-runtime</button>
    </div>
  ),
}));

vi.mock("../EvidencePanel", () => ({
  EvidencePanel: (props: any) => (
    <div data-testid="evidence-panel">
      <div>route:{props.cfg?.route}</div>
      <button onClick={props.onRefresh}>refresh-logs</button>
      <button onClick={() => props.onChangeCfg({ ...props.cfg, route: "/v1/custom", limit: 10 })}>
        set-evidence-cfg
      </button>
    </div>
  ),
}));

import { DemoPage } from "../DemoPage";
import * as api from "../../../lib/api";
import {
  makeAdminLogsPage,
  makeAdminPolicySnapshot,
  makeModelsResponse,
} from "../../../test/factories/api";

function mockDefaults() {
  (api.getApiBaseUrl as any).mockReturnValue("/api");
  (api.listModels as any).mockResolvedValue(
    makeModelsResponse({
      default_model: "m1",
      deployment_capabilities: { generate: true, extract: false },
    })
  );
  (api.adminGetPolicy as any).mockResolvedValue(
    makeAdminPolicySnapshot({ status: "allow", ok: true, enable_extract: true })
  );
  (api.adminListLogs as any).mockResolvedValue(makeAdminLogsPage());
  (api.adminWriteGenerateSlo as any).mockResolvedValue({ ok: true });
  (api.adminReloadPolicy as any).mockResolvedValue({ ok: true });
  (api.adminReloadRuntime as any).mockResolvedValue({ ok: true });
}

describe("DemoPage integration", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockDefaults();
  });

  it("initial load refreshes models/policy/logs with default evidence config", async () => {
    render(<DemoPage />);

    await waitFor(() => expect(api.listModels).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.adminGetPolicy).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.adminListLogs).toHaveBeenCalledTimes(1));

    expect(api.adminListLogs).toHaveBeenCalledWith(
      expect.objectContaining({
        route: "/v1/generate",
        limit: 50,
        offset: 0,
      })
    );
    expect(screen.getByText("model:m1")).toBeInTheDocument();
    expect(screen.getByText("effective:false")).toBeInTheDocument();
  });

  it("track switch resets evidence route and refresh-now uses updated config", async () => {
    render(<DemoPage />);
    await waitFor(() => expect(api.adminListLogs).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole("button", { name: "switch-track" }));
    fireEvent.click(screen.getByRole("button", { name: "refresh-now" }));

    await waitFor(() => expect(api.adminListLogs).toHaveBeenCalledTimes(2));
    expect(api.adminListLogs).toHaveBeenLastCalledWith(
      expect.objectContaining({
        route: "/v1/extract",
        limit: 50,
        offset: 0,
      })
    );
    expect(screen.getByText("route:/v1/extract")).toBeInTheDocument();
  });

  it("write-slo and reload actions show success banner and trigger refreshes", async () => {
    render(<DemoPage />);
    await waitFor(() => expect(api.listModels).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.adminGetPolicy).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.adminListLogs).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole("button", { name: "write-slo" }));
    await waitFor(() => expect(api.adminWriteGenerateSlo).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.listModels).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(api.adminGetPolicy).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(api.adminListLogs).toHaveBeenCalledTimes(2));
    expect(screen.getByText(/Wrote generate SLO snapshot/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "reload-policy" }));
    await waitFor(() => expect(api.adminReloadPolicy).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.adminGetPolicy).toHaveBeenCalledTimes(3));
    expect(screen.getByText(/Reloaded policy snapshot/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "reload-runtime" }));
    await waitFor(() => expect(api.adminReloadRuntime).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.listModels).toHaveBeenCalledTimes(3));
    await waitFor(() => expect(api.adminGetPolicy).toHaveBeenCalledTimes(4));
    await waitFor(() => expect(api.adminListLogs).toHaveBeenCalledTimes(3));
    expect(screen.getByText(/Reloaded runtime/i)).toBeInTheDocument();
  });

  it("shows error banner when action fails", async () => {
    (api.adminReloadPolicy as any).mockRejectedValueOnce(new Error("reload failed"));
    render(<DemoPage />);
    await waitFor(() => expect(api.adminGetPolicy).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole("button", { name: "reload-policy" }));
    expect(await screen.findByText(/Error: Error: reload failed/i)).toBeInTheDocument();
  });

  it("auto-refresh interval triggers refreshAll", async () => {
    vi.useFakeTimers();
    render(<DemoPage />);
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(api.listModels).toHaveBeenCalledTimes(1);
    expect(api.adminGetPolicy).toHaveBeenCalledTimes(1);
    expect(api.adminListLogs).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("button", { name: "set-refresh-2s" }));
    fireEvent.click(screen.getByRole("button", { name: "toggle-auto-refresh" }));

    await act(async () => {
      vi.advanceTimersByTime(2100);
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(api.listModels).toHaveBeenCalledTimes(2);
    expect(api.adminGetPolicy).toHaveBeenCalledTimes(2);
    expect(api.adminListLogs).toHaveBeenCalledTimes(2);
    vi.useRealTimers();
  });
});
