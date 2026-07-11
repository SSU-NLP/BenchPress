// Per-vendor colors so model rows are visually grouped by company.
// Curated hues for the major labs; everything else gets a stable hash hue.

const CURATED_HUE: Record<string, number> = {
  OpenAI: 162, // teal-green
  Anthropic: 24, // clay
  Google: 222, // blue
  DeepMind: 222,
  Alibaba: 265, // Qwen purple
  Qwen: 265,
  DeepSeek: 234, // indigo
  Moonshot: 288, // Kimi violet
  Meta: 205, // azure
  "Z.ai": 188, // cyan
  Zhipu: 188,
  Mistral: 40, // amber
  xAI: 280,
  NVIDIA: 96, // green
  Microsoft: 210,
  Baidu: 350, // red
  SKT: 330, // pink
  LG: 318, // magenta
  Cohere: 320,
  "01.AI": 145,
  Tencent: 200,
  ByteDance: 348,
  Upstage: 12,
};

// Vendors rendered near-neutral (no strong brand hue).
const LOW_SAT = new Set<string>(["xAI"]);

function hashHue(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return h % 360;
}

function hueFor(vendor: string): number {
  return CURATED_HUE[vendor] ?? hashHue(vendor);
}

/** Light row-background tint for a vendor. */
export function vendorBg(vendor: string): string {
  const h = hueFor(vendor);
  const s = LOW_SAT.has(vendor) ? 10 : 62;
  return `hsl(${h} ${s}% 93%)`;
}

/** Slightly stronger version for the sticky model cell / swatches. */
export function vendorSwatch(vendor: string): string {
  const h = hueFor(vendor);
  const s = LOW_SAT.has(vendor) ? 12 : 70;
  return `hsl(${h} ${s}% 62%)`;
}
