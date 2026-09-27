import { createI18n } from 'vue-i18n'
import languages from '../../../locales/languages.json'

const localeFiles = import.meta.glob('../../../locales/!(languages).json', { eager: true })

const messages = {}

for (const path in localeFiles) {
  const key = path.match(/\/([^/]+)\.json$/)[1]
  if (languages[key]) {
    messages[key] = localeFiles[path].default
  }
}

const savedLocale = 'en'
try {
  localStorage.setItem('locale', 'en')
} catch (e) {}

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  fallbackLocale: 'en',
  messages
})

const availableLocales = [{ key: 'en', label: 'English' }]

export { availableLocales }
export default i18n
