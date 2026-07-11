/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{astro,html,js,jsx,ts,tsx,md,mdx}", "./demo/**/*.{html,js,jsx,ts,tsx}"],
  theme: {
    extend: {
      // Site base font is Geist Mono (loaded in demo/index.html); both the
      // sans and mono slots point at it so every utility resolves to Geist Mono,
      // falling back to the platform monospace stack.
      fontFamily: {
        sans: [
          '"Geist Mono"',
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "monospace",
        ],
        mono: [
          '"Geist Mono"',
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },
      // Type scale on a 4px grid (sizes and line-heights are all multiples of 4).
      // 12 = notes/labels floor, 16 = body, 20 = emphasis, 24+ = titles.
      fontSize: {
        xs: ["0.75rem", "1rem"], // 12 / 16
        sm: ["1rem", "1.5rem"], // 16 / 24  (body)
        base: ["1.25rem", "1.75rem"], // 20 / 28
        lg: ["1.5rem", "2rem"], // 24 / 32  (section title)
        xl: ["1.75rem", "2.25rem"], // 28 / 36
        "2xl": ["2rem", "2.5rem"], // 32 / 40  (page title)
        "3xl": ["2.25rem", "2.75rem"], // 36 / 44
        "4xl": ["2.75rem", "3.25rem"], // 44 / 52 (hero)
      },
    },
  },
  plugins: [],
};
