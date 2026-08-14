<template>
  <div class="reports-view">
    <el-row :gutter="16">
      <!-- ============================ 左栏：查询 + 摘要 ============================ -->
      <el-col :span="10">
        <el-card>
          <template #header><span>报告查询</span></template>
          <el-form label-width="70px" size="small">
            <el-form-item label="作业 ID">
              <el-input
                v-model="jobId"
                placeholder="输入 job_id"
                clearable
                @keyup.enter="loadReport"
              />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :icon="Search" :loading="loadingReport" @click="loadReport">
                查询报告
              </el-button>
              <el-button v-if="reportHtml" :icon="View" @click="reportDialogVisible = true">
                大图查看
              </el-button>
            </el-form-item>
          </el-form>

          <div class="recent-jobs">
            <div class="recent-title">最近作业（点击选择）</div>
            <el-select
              v-model="jobId"
              filterable
              clearable
              placeholder="选择最近作业"
              style="width: 100%"
              @change="onJobPicked"
            >
              <el-option
                v-for="j in recentJobs"
                :key="j.job_id"
                :label="`${j.name} — ${j.job_id}`"
                :value="j.job_id"
              />
            </el-select>
          </div>
        </el-card>

        <el-card v-if="summary" class="summary-card">
          <template #header><span>报告摘要</span></template>
          <el-descriptions :column="1" border size="small">
            <el-descriptions-item label="作业名称">{{ summary.name }}</el-descriptions-item>
            <el-descriptions-item label="作业类型">{{ summary.job_type }}</el-descriptions-item>
            <el-descriptions-item label="状态">
              <el-tag :type="statusTagType(summary.status)" disable-transitions>
                {{ statusLabel(summary.status) }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="创建时间">{{ formatTime(summary.created_at) }}</el-descriptions-item>
            <el-descriptions-item label="诊断快照">{{ summary.diagnostic_steps }}</el-descriptions-item>
            <el-descriptions-item label="力系数行数">{{ summary.force_rows }}</el-descriptions-item>
            <el-descriptions-item label="结果图片">{{ summary.image_count }}</el-descriptions-item>
          </el-descriptions>

          <div class="kpi-grid">
            <div
              v-for="item in kpiItems"
              :key="item.label"
              class="kpi-item"
            >
              <div class="kpi-label">{{ item.label }}</div>
              <div class="kpi-value">{{ item.value }}</div>
            </div>
          </div>

          <el-alert
            v-if="summary.error"
            :title="`作业错误：${summary.error}`"
            type="error"
            :closable="false"
            class="error-alert"
          />
        </el-card>
      </el-col>

      <!-- ============================ 右栏：报告预览 + KPI 对比 ============================ -->
      <el-col :span="14">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>HTML 报告预览</span>
              <el-button
                v-if="reportHtml"
                size="small"
                :icon="Refresh"
                @click="loadReport"
              >
                重新加载
              </el-button>
            </div>
          </template>
          <iframe v-if="reportHtml" :srcdoc="reportHtml" class="report-frame" title="HTML 报告" />
          <el-empty v-else description="查询作业后在此预览 HTML 报告" />
        </el-card>

        <el-card class="compare-card">
          <template #header><span>KPI 对比（多作业）</span></template>
          <div class="compare-input">
            <el-input
              v-model="compareIdsText"
              type="textarea"
              :rows="3"
              placeholder="输入多个 job_id，以逗号 / 空格 / 换行分隔（至少 2 个）"
            />
            <el-button
              type="primary"
              :icon="DataAnalysis"
              :loading="loadingCompare"
              class="compare-btn"
              @click="loadCompare"
            >
              对比 KPI
            </el-button>
          </div>

          <el-alert
            v-if="compareResult && compareResult.missing.length"
            :title="`以下作业未找到：${compareResult.missing.join('、')}`"
            type="warning"
            :closable="false"
            class="missing-alert"
          />

          <template v-if="compareResult && compareResult.rows.length">
            <el-table
              :data="compareResult.rows"
              border
              size="small"
              class="compare-table"
              max-height="360"
            >
              <el-table-column prop="name" label="作业" min-width="140" fixed show-overflow-tooltip />
              <el-table-column prop="job_id" label="ID" width="110" show-overflow-tooltip />
              <el-table-column label="状态" width="90">
                <template #default="{ row }">
                  <el-tag :type="statusTagType(row.status)" size="small" disable-transitions>
                    {{ statusLabel(row.status) }}
                  </el-tag>
                </template>
              </el-table-column>
              <el-table-column
                v-for="m in metricKeys"
                :key="m"
                :label="m"
                min-width="110"
                align="center"
              >
                <template #default="{ row }">{{ formatMetric(row.compare_metrics[m]) }}</template>
              </el-table-column>
            </el-table>

            <el-divider content-position="left">指标统计</el-divider>
            <el-table :data="metricSummaryRows" border size="small" max-height="300">
              <el-table-column prop="metric" label="指标" min-width="150" show-overflow-tooltip />
              <el-table-column label="最小值" align="center">
                <template #default="{ row }">{{ formatMetric(row.min) }}</template>
              </el-table-column>
              <el-table-column label="最大值" align="center">
                <template #default="{ row }">{{ formatMetric(row.max) }}</template>
              </el-table-column>
              <el-table-column label="均值" align="center">
                <template #default="{ row }">{{ formatMetric(row.mean) }}</template>
              </el-table-column>
              <el-table-column prop="best_job_id" label="最优作业" min-width="140" show-overflow-tooltip />
            </el-table>
          </template>
        </el-card>
      </el-col>
    </el-row>

    <!-- ============================ 报告大图弹窗 ============================ -->
    <el-dialog
      v-model="reportDialogVisible"
      title="HTML 报告"
      width="90%"
      top="3vh"
      destroy-on-close
    >
      <iframe v-if="reportHtml" :srcdoc="reportHtml" class="report-frame-dialog" title="HTML 报告" />
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { DataAnalysis, Refresh, Search, View } from '@element-plus/icons-vue'
import { listJobs, type Job } from '@/api/production'
import {
  compareReportsKpis,
  getReportHtml,
  getReportSummary,
  type CompareResponse,
  type ReportSummary,
} from '@/api/reports'

// ---------------------------------------------------------------------------
// 展示辅助
// ---------------------------------------------------------------------------

function formatTime(iso: string | null): string {
  if (!iso) return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString()
}

function formatMetric(value: number | undefined | null): string {
  if (value === undefined || value === null) return '—'
  if (typeof value === 'boolean') return value ? '是' : '否'
  return Number(value).toPrecision(4)
}

function statusLabel(status: string): string {
  const map: Record<string, string> = {
    queued: '排队中',
    running: '运行中',
    completed: '已完成',
    failed: '失败',
    cancelled: '已取消',
  }
  return map[status] || status
}

function statusTagType(status: string): 'success' | 'warning' | 'danger' | 'info' | 'primary' {
  const map: Record<string, 'success' | 'warning' | 'danger' | 'info' | 'primary'> = {
    queued: 'info',
    running: 'warning',
    completed: 'success',
    failed: 'danger',
    cancelled: 'info',
  }
  return map[status] || 'primary'
}

// ---------------------------------------------------------------------------
// 报告查询
// ---------------------------------------------------------------------------

const jobId = ref('')
const loadingReport = ref(false)
const summary = ref<ReportSummary | null>(null)
const reportHtml = ref('')
const reportDialogVisible = ref(false)

async function loadReport() {
  const id = jobId.value.trim()
  if (!id) {
    ElMessage.warning('请输入作业 ID')
    return
  }
  loadingReport.value = true
  reportHtml.value = ''
  summary.value = null
  try {
    const [sum, html] = await Promise.all([getReportSummary(id), getReportHtml(id)])
    summary.value = sum
    reportHtml.value = html
  } catch {
    // 错误提示由 request 拦截器统一处理
  } finally {
    loadingReport.value = false
  }
}

function onJobPicked() {
  // 仅在非空选择时触发查询，避免清空下拉时弹出“请输入作业 ID”提示
  if (jobId.value) loadReport()
}

// ---------------------------------------------------------------------------
// 最近作业
// ---------------------------------------------------------------------------

const recentJobs = ref<Job[]>([])

async function loadRecentJobs() {
  try {
    const res = await listJobs({ limit: 50 })
    recentJobs.value = (res.jobs ?? []).slice(0, 20)
  } catch {
    recentJobs.value = []
  }
}

// ---------------------------------------------------------------------------
// 摘要 KPI 展示
// ---------------------------------------------------------------------------

const kpiItems = computed(() => {
  const k = summary.value?.engineering_kpis
  if (!k) return []
  return [
    { label: '最新步数', value: k.latest_step ?? '—' },
    { label: '运行时长 (s)', value: k.runtime_seconds ?? '—' },
    { label: '尾段均值 Cd', value: k.mean_cd_last === null ? '—' : Number(k.mean_cd_last).toPrecision(4) },
    { label: '尾段均值 Cl', value: k.mean_cl_last === null ? '—' : Number(k.mean_cl_last).toPrecision(4) },
    {
      label: '稳态得分',
      value: k.steady_state_score === null ? '—' : Number(k.steady_state_score).toPrecision(4),
    },
    { label: '稳态判定', value: k.steady_state_detected ? '是' : '否' },
  ]
})

// ---------------------------------------------------------------------------
// KPI 对比
// ---------------------------------------------------------------------------

const compareIdsText = ref('')
const loadingCompare = ref(false)
const compareResult = ref<CompareResponse | null>(null)

const metricKeys = computed<string[]>(() => {
  const rows = compareResult.value?.rows ?? []
  const keys = new Set<string>()
  for (const row of rows) {
    for (const key of Object.keys(row.compare_metrics ?? {})) keys.add(key)
  }
  return Array.from(keys).sort()
})

const metricSummaryRows = computed(() => {
  const summaryMap = compareResult.value?.metric_summary ?? {}
  return Object.entries(summaryMap).map(([metric, stat]) => ({
    metric,
    min: stat.min,
    max: stat.max,
    mean: stat.mean,
    best_job_id: stat.best_job_id,
  }))
})

function parseJobIds(text: string): string[] {
  return text
    .split(/[,，;；\s]+/)
    .map((s) => s.trim())
    .filter(Boolean)
}

async function loadCompare() {
  const ids = parseJobIds(compareIdsText.value)
  if (ids.length < 2) {
    ElMessage.warning('请输入至少 2 个作业 ID')
    return
  }
  loadingCompare.value = true
  try {
    compareResult.value = await compareReportsKpis(ids)
  } catch {
    // 错误提示由拦截器处理
  } finally {
    loadingCompare.value = false
  }
}

onMounted(loadRecentJobs)
</script>

<style scoped>
.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.recent-jobs {
  margin-top: 4px;
}
.recent-title {
  color: #909399;
  font-size: 13px;
  margin-bottom: 8px;
}
.summary-card {
  margin-top: 16px;
}
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  margin-top: 14px;
}
.kpi-item {
  background: #f5f7fa;
  border-radius: 6px;
  padding: 8px 10px;
}
.kpi-label {
  color: #909399;
  font-size: 12px;
}
.kpi-value {
  font-size: 15px;
  font-weight: 600;
  margin-top: 2px;
}
.error-alert {
  margin-top: 12px;
}
.report-frame {
  width: 100%;
  height: 560px;
  border: 1px solid #e4e7ed;
  border-radius: 6px;
}
.report-frame-dialog {
  width: 100%;
  height: 78vh;
  border: 1px solid #e4e7ed;
  border-radius: 6px;
}
.compare-card {
  margin-top: 16px;
}
.compare-input {
  display: flex;
  align-items: flex-start;
  gap: 10px;
}
.compare-btn {
  flex-shrink: 0;
}
.missing-alert {
  margin-top: 10px;
}
.compare-table {
  margin-top: 12px;
}
</style>
