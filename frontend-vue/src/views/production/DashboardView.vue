<template>
  <div>
    <!-- 概览统计卡片 -->
    <el-row :gutter="16" class="stats-row">
      <el-col :span="5" v-for="s in summaryCards" :key="s.key">
        <el-card class="stat-card" shadow="hover">
          <div class="stat-label">{{ s.label }}</div>
          <div class="stat-value" :style="{ color: s.color }">{{ s.value }}</div>
        </el-card>
      </el-col>
    </el-row>

    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ t('production.dashboard.title') }}</span>
          <div class="header-actions">
            <el-select
              v-model="statusFilter"
              :placeholder="t('production.dashboard.allStatuses')"
              clearable
              style="width: 150px"
              @change="loadJobs"
            >
              <el-option :label="t('production.status.queued')" value="queued" />
              <el-option :label="t('production.status.running')" value="running" />
              <el-option :label="t('production.status.completed')" value="completed" />
              <el-option :label="t('production.status.failed')" value="failed" />
              <el-option :label="t('production.status.cancelled')" value="cancelled" />
            </el-select>
            <el-switch v-model="autoRefresh" :active-text="t('production.dashboard.autoRefresh')" />
            <el-button type="primary" :icon="Refresh" :loading="loading" @click="loadJobs">
              {{ t('production.common.refresh') }}
            </el-button>
          </div>
        </div>
      </template>

      <el-table :data="jobs" v-loading="loading" stripe style="width: 100%">
        <el-table-column prop="name" :label="t('production.common.name')" min-width="200" show-overflow-tooltip />
        <el-table-column prop="job_id" :label="t('production.common.jobId')" width="120">
          <template #default="{ row }">
            <el-text class="mono" size="small">{{ row.job_id }}</el-text>
          </template>
        </el-table-column>
        <el-table-column prop="job_type" :label="t('production.common.type')" width="160" show-overflow-tooltip />
        <el-table-column :label="t('production.common.status')" width="110">
          <template #default="{ row }">
            <el-tag :type="statusTagType(row.status)" disable-transitions>{{ statusLabel(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="t('production.dashboard.resources')" width="120">
          <template #default="{ row }">
            <span class="muted">{{ row.assigned_resource || '—' }}</span>
          </template>
        </el-table-column>
        <el-table-column :label="t('production.common.duration')" width="110">
          <template #default="{ row }">
            <span class="muted">{{ formatDuration(row.total_duration_seconds) }}</span>
          </template>
        </el-table-column>
        <el-table-column :label="t('production.common.createdAt')" width="190">
          <template #default="{ row }">
            <span class="muted">{{ formatTime(row.created_at) }}</span>
          </template>
        </el-table-column>
        <el-table-column :label="t('production.common.action')" width="160" fixed="right">
          <template #default="{ row }">
            <el-button
              v-if="['queued', 'running'].includes(row.status)"
              size="small"
              type="warning"
              @click="onCancel(row)"
            >
              {{ t('production.common.cancel') }}
            </el-button>
            <el-button
              v-if="['completed', 'failed', 'cancelled'].includes(row.status)"
              size="small"
              type="danger"
              plain
              @click="onDelete(row)"
            >
              {{ t('production.common.delete') }}
            </el-button>
          </template>
        </el-table-column>
        <template #empty>
          <el-empty :description="t('production.common.noJobs')" />
        </template>
      </el-table>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Refresh } from '@element-plus/icons-vue'
import { useI18n } from 'vue-i18n'
import { cancelJob, deleteJob, listJobs, type Job } from '@/api/production'

const { t } = useI18n()

const jobs = ref<Job[]>([])
const loading = ref(false)
const statusFilter = ref('')
const autoRefresh = ref(true)

let timer: ReturnType<typeof setInterval> | null = null

const summaryCards = computed(() => {
  const total = jobs.value.length
  const running = jobs.value.filter((j) => j.status === 'running').length
  const completed = jobs.value.filter((j) => j.status === 'completed').length
  const failed = jobs.value.filter((j) => j.status === 'failed').length
  const queued = jobs.value.filter((j) => j.status === 'queued').length
  return [
    { key: 'total', label: t('production.dashboard.totalJobs'), value: total, color: '#409eff' },
    { key: 'running', label: t('production.status.running'), value: running, color: '#e6a23c' },
    { key: 'queued', label: t('production.status.queued'), value: queued, color: '#909399' },
    { key: 'completed', label: t('production.status.completed'), value: completed, color: '#67c23a' },
    { key: 'failed', label: t('production.status.failed'), value: failed, color: '#f56c6c' },
  ]
})

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

function formatDuration(seconds: number | null): string {
  if (seconds == null) return '—'
  if (seconds < 1) return '<1s'
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  return `${m}m${s}s`
}

function formatTime(iso: string | null): string {
  if (!iso) return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString()
}

async function loadJobs() {
  loading.value = true
  try {
    const res = await listJobs({
      status: statusFilter.value || undefined,
      limit: 0,
    })
    jobs.value = res.jobs
  } finally {
    loading.value = false
  }
}

async function onCancel(row: Job) {
  try {
    await ElMessageBox.confirm(t('production.common.confirmCancel', { name: row.name }), t('production.common.hint'), {
      type: 'warning',
    })
  } catch {
    return
  }
  await cancelJob(row.job_id)
  ElMessage.success(t('production.common.cancelSent'))
  await loadJobs()
}

async function onDelete(row: Job) {
  try {
    await ElMessageBox.confirm(
      t('production.dashboard.confirmDelete', { name: row.name }),
      t('production.common.warning'),
      { type: 'error' },
    )
  } catch {
    return
  }
  await deleteJob(row.job_id)
  ElMessage.success(t('production.dashboard.jobDeleted'))
  await loadJobs()
}

function startAutoRefresh() {
  stopAutoRefresh()
  timer = setInterval(() => {
    loadJobs()
  }, 5000)
}

function stopAutoRefresh() {
  if (timer) {
    clearInterval(timer)
    timer = null
  }
}

onMounted(() => {
  loadJobs()
})

watch(
  autoRefresh,
  (val) => {
    if (val) startAutoRefresh()
    else stopAutoRefresh()
  },
  { immediate: true },
)

onBeforeUnmount(() => {
  stopAutoRefresh()
})
</script>

<style scoped>
.stats-row {
  margin-bottom: 16px;
}
.stat-card {
  text-align: center;
}
.stat-label {
  font-size: 13px;
  color: #909399;
}
.stat-value {
  font-size: 28px;
  font-weight: 700;
  margin-top: 6px;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.header-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}
.mono {
  font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
}
.muted {
  color: #909399;
  font-size: 13px;
}
</style>
