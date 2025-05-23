/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
      "./templates/**/*.html",
      "./assets/**/*.js"
    ],
    theme: {
      extend: {
        fontFamily: {
          mono: ['"Space Mono"', 'monospace'],
        },
      },
    },
    plugins: [require('@tailwindcss/typography'), require('tailwind-scrollbar')],
  }