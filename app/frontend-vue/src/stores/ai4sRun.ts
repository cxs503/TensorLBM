import { ref } from 'vue'
import type { RunReportOut } from '@/api/apps'

/**
 * 轻量的“最近一次运行”共享状态：在本地全栈运行成功后保存 RunReport，
 * 供血缘追溯视图（AppLineageView）展示 lineage_upstream 血缘链。
 *
 * 使用 localStorage 持久化，页面刷新后仍可追溯最近一次运行的血缘。
 */
const STORAGE_KEY = 'tensorlbm.ai4s.lastRunReport'

function loadLastReport(): RunReportOut | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? (JSON.parse(raw) as RunReportOut) : null
  } catch {
    return null
  }
}

const lastReport = ref<RunReportOut | null>(loadLastReport())

export function useRunHistory() {
  /** 保存最近一次运行报告（同时写入 localStorage）。 */
  function save(report: RunReportOut) {
    lastReport.value = report
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(report))
    } catch {
      /* localStorage 不可用时静默忽略 */
    }
  }

  return { lastReport, save }
}
