<template>
  <div>
    <el-row :gutter="16">
      <el-col :span="9">
        <el-card>
          <template #header><span>{{ t('production.benchmarks.title') }}</span></template>
          <el-form label-width="130px" size="small">
            <el-form-item :label="t('production.benchmarks.benchType')">
              <el-select v-model="benchType" style="width: 100%" @change="onTypeChange">
                <el-option :label="t('production.benchmarks.marine')" value="marine" />
                <el-option :label="t('production.benchmarks.accuracy')" value="accuracy" />
                <el-option :label="t('production.benchmarks.ghia')" value="ghia" />
                <el-option :label="t('production.benchmarks.mlups')" value="mlups" />
                <el-option :label="t('production.benchmarks.multiphase')" value="multiphase" />
                <el-option :label="t('production.benchmarks.porous')" value="porous" />
              </el-select>
            </el-form-item>

            <template v-if="benchType === 'marine' || benchType === 'accuracy'">
              <el-form-item :label="t('production.benchmarks.testCases')">
                <el-select v-model="cases" multiple collapse-tags style="width: 100%">
                  <el-option v-for="c in caseOptions" :key="c.value" :label="c.label" :value="c.value" />
                </el-select>
              </el-form-item>
            </template>

            <template v-if="benchType === 'ghia'">
              <el-form-item :label="t('production.benchmarks.meshSize')">
                <el-input-number v-model="ghiaForm.nx" :min="16" controls-position="right" style="width: 100%" />
              </el-form-item>
              <el-form-item :label="t('production.benchmarks.reynolds')">
                <el-select v-model="ghiaForm.re" style="width: 100%">
                  <el-option label="100" :value="100" />
                  <el-option label="400" :value="400" />
                  <el-option label="1000" :value="1000" />
                </el-select>
              </el-form-item>
              <el-form-item :label="t('production.common.timeSteps')">
                <el-input-number v-model="ghiaForm.n_steps" :min="1" controls-position="right" style="width: 100%" />
              </el-form-item>
            </template>

            <template v-if="benchType === 'mlups'">
              <el-form-item :label="t('production.benchmarks.meshSizeList')">
                <el-input v-model="mlupsForm.sizesText" :placeholder="t('production.benchmarks.sizesPlaceholder')" />
              </el-form-item>
              <el-form-item :label="t('production.benchmarks.stepsPerSize')">
                <el-input-number v-model="mlupsForm.steps" :min="10" controls-position="right" style="width: 100%" />
              </el-form-item>
            </template>

            <el-form-item v-if="['marine', 'accuracy', 'multiphase', 'porous'].includes(benchType)" :label="t('production.benchmarks.fastMode')">
              <el-switch v-model="fastMode" :active-text="t('production.benchmarks.reduceSteps')" />
            </el-form-item>

            <el-form-item :label="t('production.common.device')">
              <el-select v-model="device" style="width: 100%">
                <el-option label="cpu" value="cpu" />
                <el-option label="cuda:0" value="cuda:0" />
              </el-select>
            </el-form-item>

            <el-form-item>
              <el-button type="primary" :loading="submitting" @click="submitBench">
                {{ t('production.benchmarks.submitBench') }}
              </el-button>
            </el-form-item>
          </el-form>
        </el-card>

        <el-card class="gates-card">
          <template #header><span>{{ t('production.benchmarks.gatesTitle') }}</span></template>
          <div v-for="(gate, name) in gates" :key="name" class="gate-item">
            <div class="gate-title">{{ name }} — {{ (gate as any).scenario }}</div>
            <div class="gate-desc">{{ (gate as any).description }}</div>
          </div>
          <el-empty v-if="!Object.keys(gates).length" :description="t('production.benchmarks.loading')" :image-size="50" />
        </el-card>
      </el-col>

      <el-col :span="15">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>{{ t('production.benchmarks.monitorTitle') }}</span>
              <el-button size="small" :icon="Refresh" @click="loadJobs">{{ t('production.common.refresh') }}</el-button>
            </div>
          </template>

          <el-alert
            v-if="activeJob"
            :title="t('production.benchmarks.activeJobTitle', { id: activeJob.job_id, status: statusLabel(activeJob.status) })"
            :type="statusAlertType(activeJob.status)"
            :closable="false"
            class="active-job-alert"
          >
            <template #default>
              <div>{{ t('production.benchmarks.nameLabel', { name: activeJob.name }) }}</div>
              <div v-if="activeJob.error" class="error-text">{{ activeJob.error }}</div>
            </template>
          </el-alert>

          <!-- 精度回归报告 -->
          <template v-if="activeJob && activeJob.status === 'completed' && benchType === 'accuracy'">
            <el-button size="small" type="primary" @click="loadAccuracyReport">{{ t('production.benchmarks.generateReport') }}</el-button>
            <el-descriptions v-if="accuracyReport" :column="3" border size="small" class="report-block">
              <el-descriptions-item :label="t('production.benchmarks.profile')">{{ accuracyReport.profile }}</el-descriptions-item>
              <el-descriptions-item :label="t('production.benchmarks.passRate')">{{ (accuracyReport.gate?.pass_rate ?? 0) }}</el-descriptions-item>
              <el-descriptions-item :label="t('production.benchmarks.gatePassed')">{{ accuracyReport.gate?.gate_passed ? t('production.common.yes') : t('production.common.no') }}</el-descriptions-item>
              <el-descriptions-item :label="t('production.benchmarks.checksPassed')">{{ accuracyReport.gate?.checks_passed }}</el-descriptions-item>
              <el-descriptions-item :label="t('production.benchmarks.checksTotal')">{{ accuracyReport.gate?.checks_total }}</el-descriptions-item>
              <el-descriptions-item :label="t('production.benchmarks.runtime')">{{ accuracyReport.runtime_seconds }}s</el-descriptions-item>
            </el-descriptions>
          </template>

          <el-table :data="benchJobs" v-loading="jobsLoading" stripe size="small" class="jobs-table">
            <el-table-column prop="name" :label="t('production.common.name')" min-width="200" show-overflow-tooltip />
            <el-table-column prop="job_id" :label="t('production.common.id')" width="100" />
            <el-table-column :label="t('production.common.status')" width="100">
              <template #default="{ row }">
                <el-tag :type="statusTagType(row.status)" disable-transitions>{{ statusLabel(row.status) }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column :label="t('production.common.createdAt')" width="170">
              <template #default="{ row }">
                <span class="muted">{{ formatTime(row.created_at) }}</span>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { Refresh } from '@element-plus/icons-vue'
import { useI18n } from 'vue-i18n'
import {
  benchmarkAcceptanceGates,
  benchmarkAccuracyReport,
  benchmarkAccuracyBaselines,
  getJob,
  listJobs,
  submitBenchmark,
  type Job,
} from '@/api/production'

const { t } = useI18n()

const benchType = ref('marine')
const cases = ref<string[]>(['cylinder', 'sloshing', 'pipeline', 'turbulent_channel', 'wigley', 'suboff', 'geometry_library'])
const fastMode = ref(true)
const device = ref('cpu')

const caseOptions = computed(() =>
  benchType.value === 'accuracy'
    ? [
        { value: 'cavity', label: t('production.benchmarks.caseCavity') },
        { value: 'bfs', label: t('production.benchmarks.caseBfs') },
        { value: 'rotating_cylinder', label: t('production.benchmarks.caseRotatingCylinder') },
      ]
    : [
        { value: 'cylinder', label: t('production.benchmarks.caseCylinder') },
        { value: 'sloshing', label: t('production.benchmarks.caseSloshing') },
        { value: 'pipeline', label: t('production.benchmarks.casePipeline') },
        { value: 'turbulent_channel', label: t('production.benchmarks.caseTurbulentChannel') },
        { value: 'wigley', label: t('production.benchmarks.caseWigley') },
        { value: 'suboff', label: t('production.benchmarks.caseSuboff') },
        { value: 'geometry_library', label: t('production.benchmarks.caseGeometryLibrary') },
      ],
)

const ghiaForm = reactive({ nx: 64, re: 100 as number, n_steps: 5000 })
const mlupsForm = reactive({ sizesText: '128,256,512', steps: 100 })

const submitting = ref(false)
const activeJob = ref<Job | null>(null)
const activeJobId = ref('')
const accuracyReport = ref<Record<string, any> | null>(null)

const benchJobs = ref<Job[]>([])
const jobsLoading = ref(false)
const gates = ref<Record<string, any>>({})

let pollTimer: ReturnType<typeof setInterval> | null = null

function onTypeChange() {
  accuracyReport.value = null
  if (benchType.value === 'accuracy') {
    cases.value = ['cavity', 'bfs', 'rotating_cylinder']
  } else if (benchType.value === 'marine') {
    cases.value = ['cylinder', 'sloshing', 'pipeline', 'turbulent_channel', 'wigley', 'suboff', 'geometry_library']
  }
}

function buildPayload(): Record<string, any> {
  switch (benchType.value) {
    case 'marine':
      return { cases: cases.value, fast: fastMode.value, device: device.value }
    case 'accuracy':
      return { cases: cases.value, fast: fastMode.value, device: device.value }
    case 'ghia':
      return { nx: ghiaForm.nx, re: ghiaForm.re, n_steps: ghiaForm.n_steps, device: device.value }
    case 'mlups': {
      const sizes = mlupsForm.sizesText
        .split(',')
        .map((s) => parseInt(s.trim(), 10))
        .filter((n) => !Number.isNaN(n))
      return { sizes, steps: mlupsForm.steps, device: device.value }
    }
    case 'multiphase':
      return { fast: fastMode.value, device: device.value }
    case 'porous':
      return { fast: fastMode.value, device: device.value }
    default:
      return { device: device.value }
  }
}

async function submitBench() {
  submitting.value = true
  try {
    const endpoint = `/benchmarks/${benchType.value}`
    const res = await submitBenchmark(endpoint, buildPayload())
    activeJobId.value = res.job_id
    accuracyReport.value = null
    ElMessage.success(t('production.benchmarks.benchSubmittedMsg', { id: res.job_id }))
    await refreshActiveJob()
    startPolling()
    await loadJobs()
  } finally {
    submitting.value = false
  }
}

async function refreshActiveJob() {
  if (!activeJobId.value) return
  try {
    const job = await getJob(activeJobId.value)
    activeJob.value = job
    if (['completed', 'failed', 'cancelled'].includes(job.status)) {
      stopPolling()
    }
  } catch {
    stopPolling()
  }
}

function startPolling() {
  stopPolling()
  pollTimer = setInterval(() => refreshActiveJob(), 3000)
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

async function loadAccuracyReport() {
  if (!activeJobId.value) return
  try {
    accuracyReport.value = await benchmarkAccuracyReport(activeJobId.value)
  } catch {
    // 非精度基准或未完成时忽略
  }
}

async function loadJobs() {
  jobsLoading.value = true
  try {
    const res = await listJobs({ limit: 0 })
    benchJobs.value = res.jobs.filter((j) => j.job_type.startsWith('benchmark'))
  } finally {
    jobsLoading.value = false
  }
}

async function loadGates() {
  try {
    const r = await benchmarkAcceptanceGates()
    gates.value = (r.gates as Record<string, any>) || r
  } catch {
    gates.value = {}
  }
}

function statusLabel(status: string): string {
  const map: Record<string, string> = {
    queued: t('production.status.queued'),
    running: t('production.status.running'),
    completed: t('production.status.completed'),
    failed: t('production.status.failed'),
    cancelled: t('production.status.cancelled'),
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
  return map[status] || 'info'
}

function statusAlertType(status: string): 'success' | 'warning' | 'error' | 'info' {
  const map: Record<string, 'success' | 'warning' | 'error' | 'info'> = {
    queued: 'info',
    running: 'warning',
    completed: 'success',
    failed: 'error',
    cancelled: 'info',
  }
  return map[status] || 'info'
}

function formatTime(iso: string | null): string {
  if (!iso) return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString()
}

onMounted(() => {
  loadJobs()
  loadGates()
  // 静默加载精度基线库（用于提示可用 profile）
  benchmarkAccuracyBaselines().catch(() => {})
})

onBeforeUnmount(() => {
  stopPolling()
})
</script>

<style scoped>
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.gates-card {
  margin-top: 16px;
}
.gate-item {
  border-bottom: 1px solid #f0f0f0;
  padding: 6px 0;
}
.gate-item:last-child {
  border-bottom: none;
}
.gate-title {
  font-weight: 600;
  font-size: 13px;
}
.gate-desc {
  color: #909399;
  font-size: 12px;
}
.active-job-alert {
  margin-bottom: 12px;
}
.error-text {
  margin-top: 8px;
  color: #f56c6c;
  white-space: pre-wrap;
  font-family: monospace;
  font-size: 12px;
}
.jobs-table {
  margin-top: 12px;
}
.report-block {
  margin: 12px 0;
}
.muted {
  color: #909399;
  font-size: 13px;
}
</style>
