/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        paper: '#f7f1e7',
        ink: '#23201b',
        muted: '#6f675c',
        line: '#ded3c4',
        green: '#0f5b45',
        grass: '#d9e8dc',
        clay: '#b6532c',
        straw: '#f0d58c',
      },
      fontFamily: {
        display: ['"Libre Baskerville"', 'Georgia', 'serif'],
        body: ['"Source Sans 3"', 'Arial', 'sans-serif'],
      },
      boxShadow: {
        soft: '0 18px 60px rgba(35, 32, 27, 0.12)',
      },
    },
  },
  plugins: [],
};
