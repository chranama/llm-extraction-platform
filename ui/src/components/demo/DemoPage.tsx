// ui/src/components/demo/DemoPage.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  adminGetPolicy,
  adminListLogs,
  adminReloadPolicy,
  adminReloadRuntime,
  adminWriteGenerateSlo,
  listModels,
  ModelsResponseBody,
  AdminLogsPage,
  getApiBaseUrl,
} from "../../lib/api";

import { Track, DemoActionsConfig, DemoEvidenceConfig, DemoPolicySnapshot } from "./types";
import { defaultActionsConfigForTrack, defaultEvidenceConfigForTrack, toErrorMessage } from "./utils";
import { TrackSelector } from "./TrackSelector";
import { StatePanel } from "./StatePanel";
import { ActionsPanel } from "./ActionsPanel";
import { EvidencePanel } from "./EvidencePanel";

const API_KEY = import.meta.env.VITE_API_KEY as string | undefined;

type AdminPolicyResponse = DemoPolicySnapshot | Record<string, any>;

export function DemoPage(): JSX.Element {
  const apiBase = getApiBaseUrl();

  const [track, setTrack] = useState<Track>("generate_clamp");

  const [autoRefresh, setAutoRefresh] = useState<boolean>(false);
  const [refreshEverySeconds, setRefreshEverySeconds] = useState<number>(5);

  const [models, setModels] = useState<ModelsResponseBody | null>(null);
  const [policy, setPolicy] = useState<AdminPolicyResponse | null>(null);
  const [logs, setLogs] = useState<AdminLogsPage | null>(null);

  const [loadingModels, setLoadingModels] = useState<boolean>(false);
  const [loadingPolicy, setLoadingPolicy] = useState<boolean>(false);
  const [loadingLogs, setLoadingLogs] = useState<boolean>(false);

  const [actionsCfg, setActionsCfg] = useState<DemoActionsConfig>(() =>
    defaultActionsConfigForTrack("generate_clamp")
  );
  const [evidenceCfg, setEvidenceCfg] = useState<DemoEvidenceConfig>(() =>
    defaultEvidenceConfigForTrack("generate_clamp")
  );

  const [lastError, setLastError] = useState<string | null>(null);
  const [lastActionMsg, setLastActionMsg] = useState<string | null>(null);

  // Guard against out-of-order responses (fast toggles / auto-refresh)
  const inflight = useRef({ models: 0, policy: 0, logs: 0 });

  // Reset defaults when track changes
  useEffect(() => {
    setActionsCfg(defaultActionsConfigForTrack(track));
    setEvidenceCfg(defaultEvidenceConfigForTrack(track));
    setLastError(null);
    setLastActionMsg(null);
  }, [track]);

  const hasKey = Boolean(API_KEY && API_KEY.trim().length > 0);

  const effectiveExtractEnabled = useMemo(() => {
    // "effective" for UI purposes:
    // - deployment_capabilities.extract means server can do extract in principle
    // - policy.enable_extract means policy allows it right now (if present)
    const dep = models?.deployment_capabilities?.extract;
    const polEnable = (policy as any)?.enable_extract;

    if (typeof dep === "boolean" && typeof polEnable === "boolean") return dep && polEnable;
    if (typeof polEnable === "boolean") return polEnable;
    if (typeof dep === "boolean") return dep;
    return null;
  }, [models, policy]);

  async function refreshModels(): Promise<void> {
    inflight.current.models += 1;
    const tag = inflight.current.models;

    setLoadingModels(true);
    setLastError(null);

    try {
      const m = await listModels();
      if (inflight.current.models !== tag) return;
      setModels(m);
    } catch (e) {
      if (inflight.current.models !== tag) return;
      setLastError(toErrorMessage(e));
    } finally {
      if (inflight.current.models === tag) setLoadingModels(false);
    }
  }

  async function refreshPolicy(): Promise<void> {
    inflight.current.policy += 1;
    const tag = inflight.current.policy;

    setLoadingPolicy(true);
    setLastError(null);

    try {
      const p = await adminGetPolicy();
      if (inflight.current.policy !== tag) return;
      setPolicy(p as any);
    } catch (e) {
      if (inflight.current.policy !== tag) return;
      setLastError(toErrorMessage(e));
    } finally {
      if (inflight.current.policy === tag) setLoadingPolicy(false);
    }
  }

  async function refreshLogs(): Promise<void> {
    inflight.current.logs += 1;
    const tag = inflight.current.logs;

    setLoadingLogs(true);
    setLastError(null);

    try {
      const page = await adminListLogs({
        route: evidenceCfg.route || undefined,
        model_id: evidenceCfg.modelId || undefined,
        limit: evidenceCfg.limit,
        offset: 0,
      });
      if (inflight.current.logs !== tag) return;
      setLogs(page);
    } catch (e) {
      if (inflight.current.logs !== tag) return;
      setLastError(toErrorMessage(e));
    } finally {
      if (inflight.current.logs === tag) setLoadingLogs(false);
    }
  }

  async function refreshAll(): Promise<void> {
    await Promise.all([refreshModels(), refreshPolicy(), refreshLogs()]);
  }

  // Initial load
  useEffect(() => {
    void refreshAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto refresh
  useEffect(() => {
    if (!autoRefresh) return;

    const ms = Math.max(2, refreshEverySeconds) * 1000;
    const t = window.setInterval(() => {
      void refreshAll();
    }, ms);

    return () => window.clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    autoRefresh,
    refreshEverySeconds,
    track,
    evidenceCfg.route,
    evidenceCfg.modelId,
    evidenceCfg.limit,
  ]);

  // Actions
  async function onWriteGenerateSlo(): Promise<void> {
    setLastActionMsg(null);
    setLastError(null);

    try {
      await adminWriteGenerateSlo({
        window_seconds: actionsCfg.sloWindowSeconds,
        route: actionsCfg.sloRoute || undefined,
        model_id: actionsCfg.sloModelId || undefined,
        out_path: actionsCfg.sloOutPath || undefined,
      });

      setLastActionMsg(`✅ Wrote generate SLO snapshot (${actionsCfg.sloWindowSeconds}s window).`);
      // best-effort refresh (do not block UI)
      void refreshPolicy();
      void refreshLogs();
      void refreshModels();
    } catch (e) {
      setLastError(toErrorMessage(e));
    }
  }

  async function onReloadPolicy(): Promise<void> {
    setLastActionMsg(null);
    setLastError(null);

    try {
      await adminReloadPolicy();
      setLastActionMsg("✅ Reloaded policy snapshot.");
      void refreshPolicy();
    } catch (e) {
      setLastError(toErrorMessage(e));
    }
  }

  async function onReloadRuntime(): Promise<void> {
    setLastActionMsg(null);
    setLastError(null);

    try {
      await adminReloadRuntime();
      setLastActionMsg("✅ Reloaded runtime (policy + models applied).");
      void refreshAll();
    } catch (e) {
      setLastError(toErrorMessage(e));
    }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <TrackSelector
        track={track}
        onChangeTrack={setTrack}
        apiBase={apiBase}
        hasApiKey={hasKey}
        autoRefresh={autoRefresh}
        onToggleAutoRefresh={() => setAutoRefresh((v) => !v)}
        refreshEverySeconds={refreshEverySeconds}
        onChangeRefreshEverySeconds={setRefreshEverySeconds}
        onRefreshNow={() => void refreshAll()}
        isRefreshing={loadingModels || loadingPolicy || loadingLogs}
      />

      {(lastActionMsg || lastError) && (
        <div
          style={{
            borderRadius: 12,
            border: `1px solid ${lastError ? "#fed7aa" : "#bbf7d0"}`,
            background: lastError ? "#fff7ed" : "#ecfdf5",
            color: lastError ? "#9a3412" : "#065f46",
            padding: 12,
            whiteSpace: "pre-wrap",
            fontWeight: 650,
          }}
        >
          {lastError ? `Error: ${lastError}` : lastActionMsg}
        </div>
      )}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1.1fr 0.9fr 1.2fr",
          gap: 14,
          alignItems: "start",
        }}
      >
        <StatePanel
          models={models}
          policy={policy}
          effectiveExtractEnabled={effectiveExtractEnabled}
          loadingModels={loadingModels}
          loadingPolicy={loadingPolicy}
          onRefreshModels={() => void refreshModels()}
          onRefreshPolicy={() => void refreshPolicy()}
        />

        <ActionsPanel
          track={track}
          cfg={actionsCfg}
          onChangeCfg={setActionsCfg}
          onWriteGenerateSlo={() => void onWriteGenerateSlo()}
          onReloadPolicy={() => void onReloadPolicy()}
          onReloadRuntime={() => void onReloadRuntime()}
          disabled={!hasKey}
        />

        <EvidencePanel
          track={track}
          cfg={evidenceCfg}
          onChangeCfg={setEvidenceCfg}
          logs={logs}
          loading={loadingLogs}
          onRefresh={() => void refreshLogs()}
        />
      </div>
    </div>
  );
}