import React from "react";
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { PolicyCard } from "../PolicyCard";

describe("PolicyCard", () => {
  it("renders empty state when policy is null", () => {
    render(<PolicyCard policy={null} />);
    expect(screen.getByText(/No policy snapshot loaded/i)).toBeInTheDocument();
  });

  it("renders policy fields, issues, truncation and metrics", () => {
    const reasons = new Array(7).fill(0).map((_, i) => ({
      code: `r${i}`,
      message: `reason ${i}`,
      context: i === 0 ? { a: 1 } : undefined,
    }));
    const warnings = [{ code: "w1", message: "warn 1" }];

    render(
      <PolicyCard
        policy={{
          policy: "extract_enablement",
          generated_at: "2026-01-01T00:00:00Z",
          pipeline: "extract_only",
          status: "allow",
          ok: true,
          enable_extract: true,
          generate_max_new_tokens_cap: 256,
          thresholds_profile: "extract/default",
          generate_thresholds_profile: "generate/default",
          reasons,
          warnings,
          metrics: { score: 0.99 },
        }}
      />
    );

    expect(screen.getByText("extract_enablement")).toBeInTheDocument();
    expect(screen.getByText(/allow \/ true/i)).toBeInTheDocument();
    expect(screen.getByText("256")).toBeInTheDocument();
    expect(screen.getByText("r0")).toBeInTheDocument();
    expect(screen.getByText(/and 1 more/i)).toBeInTheDocument();
    expect(screen.getByText("w1")).toBeInTheDocument();
    expect(screen.getByText(/"score": 0.99/i)).toBeInTheDocument();
  });
});

