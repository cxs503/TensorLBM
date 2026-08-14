<template>
  <div>
    <el-row :gutter="16">
      <el-col :span="9">
        <el-card>
          <template #header><span>基准测试配置</span></template>
          <el-form label-width="130px" size="small">
            <el-form-item label="基准类型">
              <el-select v-model="benchType" style="width: 100%" @change="onTypeChange">
                <el-option label="Marine 海洋工程套件" value="marine" />
                <el-option label="Accuracy 精度回归" value="accuracy" />
                <el-option label="Ghia 方腔基准" value="ghia" />
                <el-option label="MLUPS 性能基准" value="mlups" />
                <el-option label="Multiphase 多相流" value="multiphase" />
                <el-option label="Porous 多孔介质" value="porous" />
              </el-select>
            </el-form-item>

            <template v-if="benchType === 'marine' || benchType === 'accuracy'">
              <el-form-item label="测试用例">
                <el-select v-model="cases" multiple collapse-tags style="width: 100%">
                  <el-option v-for="c in caseOptions" :key="c.value" :label="c.label" :value="c.value" />
                </el-select>
              </el-form-item>
            </template>

            <template v-if="benchType === 'ghia'">
              <el-form-item label="网格尺寸">
                <el-input-number v-model="ghiaForm.nx" :min="16" controls-position="right" style="width: 100%" />
              </el-form-item>
              <el-form-item label="雷诺数">
                <el-select v-model="ghiaForm.re" style="width: 100%">
                  <el-option label="100" :value="100" />
                  <el-option label="400" :value="400" />
                  <el-option label="1000" :value="1000" />
                </el-select>
              </el-form-item>
              <el-form-item label="时间步数">
                <el-input-number v-model="ghiaForm.n_steps" :min="1" controls-position="right" style="width: 100%" />
              </el-form-item>
            </template>

            <template v-if="benchType === 'mlups'">
              <el-form-item label="网格尺寸列表">
                <el-input v-model="mlupsForm.sizesText" placeholder="逗号分隔，如 128,256,512" />
              </el-form-item>
              <el-form-item label="每尺寸步数">
                <el-input-number v-model="mlupsForm.steps" :min="10" controls-position="right" style="width: 100%" />
              </el-form-item>
            </template>

            <el-form-item v-if="['marine', 'accuracy', 'multiphase', 'porous'].includes(benchType)" label="快速模式">
              <el-switch v-model="fastMode" active-text="减少步数" />
            </el-form-item>

            <el-form-item label="计算设备">
              <el-select v-model="device" style="width: 100%">
                <el-option label="cpu" value="cpu" />
                <el-option label="cuda:0" value="cuda:0" />
              </el-select>
            </el-form-item>

            <el-form-item>
              <el-button type="primary" :loading="submitting" @click="submitBench">
                提交基准测试
              </el-button>
            </el-form-item>
          </el-form>
        </el-card>

        <el-card class="gates-card">
          <template #header><span>工程验收门槛（Acceptance Gates）</span></template>
          <div v-for="(gate, name) in gates" :key="name" class="gate-item">
            <div class="gate-title">{{ name }} — {{ (gate as any).scenario }}</div>
            <div class="gate-desc">{{ (gate as any).description }}</div>
          </div>
          <el-empty v-if="!Object.keys(gates).length" description="加载中…" :image-size="50" />
        </el-card>
      </el-col>

      <el-col :span="15">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>基准运行监控</span>
              <el-button size="small" :icon="Refresh" @click="loadJobs">刷新</el-button>
            </div>
          </template>

          <el-alert
            v-if="activeJob"
            :title="`作业 ${activeJob.job_id} — ${statusLabel(activeJob.status)}`"
            :type="statusAlertType(activeJob.status)"
            :closable="false"
            class="active-job-alert"
          >
            <template #default>
              <div>名称：{{ activeJob.name }}</div>
              <div v-if="activeJob.error" class="error-text">{{ activeJob.error }}</div>
            </template>
          </el-alert>

          <!-- 精度回归报告 -->
          <template v-if="activeJob && activeJob.status === 'completed' && benchType === 'accuracy'">
            <el-button size="small" type="primary" @click="loadAccuracyReport">生成精度回归报告</el-button>
            <el-descriptions v-if="accuracyReport" :column="3" border size="small" class="report-block">
              <el-descriptions-item label="Profile">{{ accuracyReport.profile }}</el-descriptions-item>
              <el-descriptions-item label="通过率">{{ (accuracyReport.gate?.pass_rate ?? 0) }}</el-descriptions-item>
              <el-descriptions-item label="门限通过">{{ accuracyReport.gate?.gate_passed ? '是' : '否' }}</el-descriptions-item>
              <el-descriptions-item label="检查通过">{{ accuracyReport.gate?.checks_passed }}</el-descriptions-item>
              <el-descriptions-item label="检查总数">{{ accuracyReport.gate?.checks_total }}</el-descriptions-item>
              <el-descriptions-item label="运行时长">{{ accuracyReport.runtime_seconds }}s</el-descriptions-item>
            </el-descriptions>
          </template>

          <el-table :data="benchJobs" v-loading="jobsLoading" stripe size="small" class="jobs-table">
            <el-table-column prop="name" label="名称" min-width="200" show-overflow-tooltip />
            <el-table-column prop="job_id" label="ID" width="100" />
            <el-table-column label="状态" width="100">
              <template #default="{ row }">
                <el-tag :type="statusTagType(row.status)" disable-transitions>{{ statusLabel(row.status) }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column label="创建时间" width="170">
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
import {
  benchmarkAcceptanceGates,
  benchmarkAccuracyReport,
  benchmarkAccuracyBaselines,
  getJob,
  listJobs,
  submitBenchmark,
  type Job,
} from '@/api/production'

const benchType = ref('marine')
const cases = ref<string[]>(['cylinder', 'sloshing', 'pipeline', 'turbulent_channel', 'wigley', 'suboff', 'geometry_library'])
const fastMode = ref(true)
const device = ref('cpu')

const marineCases = [
  { value: 'cylinder', label: '圆柱绕流' },
  { value: 'sloshing', label: '晃荡液舱' },
  { value: 'pipeline', label: '管道流' },
  { value: 'turbulent_channel', label: '湍流通道' },
  { value: 'wigley', label: 'Wigley 船体' },
  { value: 'suboff', label: 'SUBOFF 潜艇' },
  { value: 'geometry_library', label: '几何库' },
]
const accuracyCases = [
  { value: 'cavity', label: '方腔' },
  { value: 'bfs', label: '后向台阶' },
  { value: 'rotating_cylinder', label: '旋转圆柱' },
]

const caseOptions = computed(() => (benchType.value === 'accuracy' ? accuracyCases : marineCases))

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
    ElMessage.success(`基准作业已提交：${res.job_id}`)
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
