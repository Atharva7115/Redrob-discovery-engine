/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class', // Support dark mode if needed, default to dark
  theme: {
    extend: {
      colors: {
        // Neutral base palette
        neutral: {
          50: '#F9FAFB',
          100: '#F3F4F6',
          200: '#E5E7EB',
          300: '#D1D5DB',
          400: '#9CA3AF',
          500: '#6B7280',
          600: '#4B5563',
          700: '#374151',
          800: '#1F2937',
          900: '#111827',
          950: '#030712', // Deep near-black background
        },
        // Single vibrant accent color (Electric Teal)
        accent: {
          50: '#E6FCFA',
          100: '#B3F7F2',
          200: '#80F2E9',
          300: '#4DECE1',
          400: '#26E8DC',
          500: '#00D8C9', // Core Teal Accent
          600: '#00BFA3',
          700: '#009D81',
          800: '#007B61',
          900: '#004C38',
        }
      },
      fontFamily: {
        sans: ['Inter', 'Segoe UI', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        premium: '0 4px 20px -2px rgba(0, 0, 0, 0.3)',
        accent: '0 0 15px -3px rgba(0, 216, 201, 0.4)',
      }
    },
  },
  plugins: [],
}
