import { createI18n } from 'vue-i18n'
import zh from './zh'
import en from './en'

export type Locale = 'zh' | 'en'

const STORAGE_KEY = 'tensorlbm-locale'

function getInitialLocale(): Locale {
  const saved = localStorage.getItem(STORAGE_KEY)
  if (saved === 'zh' || saved === 'en') {
    return saved
  }
  const browser = navigator.language?.toLowerCase() ?? ''
  return browser.startsWith('zh') ? 'zh' : 'en'
}

export function setLocale(locale: Locale) {
  i18n.global.locale.value = locale
  localStorage.setItem(STORAGE_KEY, locale)
}

const i18n = createI18n({
  legacy: false,
  locale: getInitialLocale(),
  fallbackLocale: 'zh',
  messages: {
    zh,
    en,
  },
})

export default i18n
