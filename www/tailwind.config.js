/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        // Serif body for blogpost feel; Archivo grotesque for UI/headings;
        // Archivo Expanded for display (hero, Part numbers); mono for numerals.
        serif:   ['"Source Serif 4"', '"Source Serif Pro"', 'Charter', 'Georgia', 'serif'],
        sans:    ['"Archivo"', 'system-ui', 'sans-serif'],
        display: ['"Archivo Expanded"', '"Archivo"', 'system-ui', 'sans-serif'],
        mono:    ['"JetBrains Mono"', '"SF Mono"', 'ui-monospace', 'monospace'],
      },
      colors: {
        ink:   { DEFAULT: '#1a1a1a', soft: '#2a2a2a', mute: '#555' },
        paper: { DEFAULT: '#fbfaf7', warm: '#f6f3ec', deep: '#efeae0' },
        accent:{ DEFAULT: '#c2410c', soft: '#fed7aa' }, // burnt orange (helix R²)
        spine: '#0369a1', // PC1 R²
        ratio: '#15803d', // helix/PCA
        rho:   '#7c3aed', // L=0 share
      },
      maxWidth: {
        // Matches the grid's `main` track in index.css, so the hero, footer,
        // figure captions, and body text all share one column width.
        prose: '820px',
      },
      typography: ({ theme }) => ({
        ink: {
          css: {
            '--tw-prose-body': theme('colors.ink.DEFAULT'),
            '--tw-prose-headings': theme('colors.ink.DEFAULT'),
          },
        },
      }),
    },
  },
  plugins: [],
};
