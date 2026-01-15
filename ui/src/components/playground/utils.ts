// ui/src/components/playground/utils.ts
export function prettyJson(x: any): string {
  try {
    return JSON.stringify(x, null, 2);
  } catch {
    return String(x);
  }
}

export function toNumberOr(prev: number, raw: string): number {
  const s = raw.trim();
  if (!s) return prev;
  const n = Number(s);
  return Number.isFinite(n) ? n : prev;
}

export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      ta.style.top = "-9999px";
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(ta);
      return ok;
    } catch {
      return false;
    }
  }
}

export function stableStringify(x: any): string {
  const seen = new WeakSet();
  const helper = (v: any): any => {
    if (v && typeof v === "object") {
      if (seen.has(v)) return "[Circular]";
      seen.add(v);
      if (Array.isArray(v)) return v.map(helper);
      const out: Record<string, any> = {};
      Object.keys(v)
        .sort()
        .forEach((k) => {
          out[k] = helper(v[k]);
        });
      return out;
    }
    return v;
  };

  try {
    return JSON.stringify(helper(x));
  } catch {
    return String(x);
  }
}

export function valueCell(v: any): string {
  if (v === undefined) return "(missing)";
  if (v === null) return "null";
  if (typeof v === "string") return v;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  return prettyJson(v);
}

export function shorten(text: string, n: number): string {
  const s = (text || "").trim();
  if (s.length <= n) return s;
  return s.slice(0, n - 3) + "...";
}