// ui/src/components/playground/__tests__/GeneratePanel.test.tsx
import React from "react";
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { GeneratePanel } from "../GeneratePanel";

function makeBaseProps(
  overrides: Partial<React.ComponentProps<typeof GeneratePanel>> = {}
): React.ComponentProps<typeof GeneratePanel> {
  return {
    loading: false,
    prompt: "hello",
    setPrompt: vi.fn(),
    genMaxNewTokens: 128,
    setGenMaxNewTokens: vi.fn(),
    genTemperature: 0.2,
    setGenTemperature: vi.fn(),
    onCopyCurl: vi.fn(),
    onRun: vi.fn(),
    activeError: null,
    genOutput: "",
    toNumberOr: (prev, raw) => {
      const s = raw.trim();
      if (!s) return prev;
      const n = Number(s);
      return Number.isFinite(n) ? n : prev;
    },
    ...overrides,
  };
}

describe("GeneratePanel", () => {
  it("renders prompt textarea + output placeholder", () => {
    const props = makeBaseProps({ prompt: "abc", genOutput: "" });
    render(<GeneratePanel {...props} />);

    expect(screen.getByText("Prompt")).toBeInTheDocument();
    expect(screen.getByDisplayValue("abc")).toBeInTheDocument();

    expect(screen.getByText("Output")).toBeInTheDocument();
    expect(screen.getByText("No output yet.")).toBeInTheDocument();
  });

  it("calls setPrompt when typing in the textarea (controlled input rerender)", async () => {
    const user = userEvent.setup();

    let prompt = "";
    const props = makeBaseProps({ prompt });

    const { rerender } = render(<GeneratePanel {...props} />);

    // make setter behave like state: update prompt + rerender
    const setPrompt = vi.fn((v: string) => {
      prompt = v;
      rerender(<GeneratePanel {...props} prompt={prompt} setPrompt={setPrompt} />);
    });

    rerender(<GeneratePanel {...props} prompt={prompt} setPrompt={setPrompt} />);

    const ta = screen.getByRole("textbox"); // textarea
    await user.type(ta, "Hi");

    expect(setPrompt).toHaveBeenCalled();
    expect(setPrompt).toHaveBeenLastCalledWith("Hi");
    expect(screen.getByDisplayValue("Hi")).toBeInTheDocument();
  });

  it("calls onRun and onCopyCurl when buttons clicked", async () => {
    const user = userEvent.setup();
    const props = makeBaseProps();

    render(<GeneratePanel {...props} />);

    await user.click(screen.getByRole("button", { name: "Generate" }));
    expect(props.onRun).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "Copy curl" }));
    expect(props.onCopyCurl).toHaveBeenCalledTimes(1);
  });

  it("disables inputs/buttons and shows running state when loading=true", () => {
    const props = makeBaseProps({ loading: true, genOutput: "" });
    render(<GeneratePanel {...props} />);

    expect(screen.getByRole("button", { name: "Running..." })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Copy curl" })).toBeDisabled();

    expect(screen.getByDisplayValue("128")).toBeDisabled();
    expect(screen.getByDisplayValue("0.2")).toBeDisabled();

    expect(screen.getByText("Waiting for response...")).toBeInTheDocument();
  });

  it("shows active error when provided", () => {
    const props = makeBaseProps({ activeError: "boom" });
    render(<GeneratePanel {...props} />);

    expect(screen.getByText(/Error: boom/)).toBeInTheDocument();
  });

  it("uses toNumberOr and calls setters when numeric inputs change (controlled input rerender)", async () => {
    let maxTokens = 128;
    let temp = 0.2;

    const toNumberOr = vi.fn((prev: number, raw: string) => {
      return raw === "999" ? 999 : prev;
    });

    const base = makeBaseProps({
      genMaxNewTokens: maxTokens,
      genTemperature: temp,
      toNumberOr,
    });

    const { rerender } = render(<GeneratePanel {...base} />);

    const setGenTemperature = vi.fn((v: number) => {
      temp = v;
      rerender(
        <GeneratePanel
          {...base}
          genMaxNewTokens={maxTokens}
          genTemperature={temp}
          setGenMaxNewTokens={setGenMaxNewTokens}
          setGenTemperature={setGenTemperature}
        />
      );
    });

    const setGenMaxNewTokens = vi.fn((v: number) => {
      maxTokens = v;
      rerender(
        <GeneratePanel
          {...base}
          genMaxNewTokens={maxTokens}
          genTemperature={temp}
          setGenMaxNewTokens={setGenMaxNewTokens}
          setGenTemperature={setGenTemperature}
        />
      );
    });

    rerender(
      <GeneratePanel
        {...base}
        genMaxNewTokens={maxTokens}
        genTemperature={temp}
        setGenMaxNewTokens={setGenMaxNewTokens}
        setGenTemperature={setGenTemperature}
      />
    );

    // max_new_tokens
    const maxTokensInput = screen.getByDisplayValue("128");
    fireEvent.change(maxTokensInput, { target: { value: "999" } });

    expect(toNumberOr).toHaveBeenCalled();
    expect(setGenMaxNewTokens).toHaveBeenCalled();
    expect(setGenMaxNewTokens).toHaveBeenLastCalledWith(999);

    // temperature
    const tempInput = screen.getByDisplayValue("0.2");
    fireEvent.change(tempInput, { target: { value: "999" } });

    expect(setGenTemperature).toHaveBeenCalled();
    expect(setGenTemperature).toHaveBeenLastCalledWith(999);
  });
});