import { createI18n } from 'vue-i18n'
import zhBase from './zh'
import enBase from './en'
import production from './modules/production'
import ai4s from './modules/ai4s'
import data from './modules/data'
import misc from './modules/misc'

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

const messages = {
  zh: {
    ...zhBase,
    production: production.zh,
    ai4s: ai4s.zh,
    data: data.zh,
    misc: misc.zh,
  },
  en: {
    ...enBase,
    production: production.en,
    ai4s: ai4s.en,
    data: data.en,
    misc: misc.en,
  },
}

const i18n = createI18n({
  legacy: false,
  locale: getInitialLocale(),
  fallbackLocale: 'zh',
  messages,
})

export default i18n
