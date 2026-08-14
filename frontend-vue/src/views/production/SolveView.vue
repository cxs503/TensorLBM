<template>
  <div>
    <el-row :gutter="16">
      <el-col :span="9">
        <el-card>
          <template #header><span>求解器配置</span></template>
          <el-form label-width="130px" size="small">
            <el-form-item label="求解类型">
              <el-select v-model="selectedType" filterable style="width: 100%" @change="onTypeChange">
                <el-option
                  v-for="(s, key) in solverTypes"
                  :key="key"
                  :label="s.label"
                  :value="key"
                />
              </el-select>
            </el-form-item>
          </el-form>
          <el-alert
            v-if="currentDesc"
            :title="currentDesc"
            type="info"
            :closable="false"
            class="desc-alert"
          />

          <el-form label-width="150px" size="small">
            <el-form-item v-for="f in currentFields" :key="f.name" :label="f.label">
              <el-input-number
                v-if="f.type === 'number'"
                v-model="formData[f.name]"
                :min="f.min"
                :max="f.max"
                :step="f.step"
                :precision="f.precision"
                controls-position="right"
                style="width: 100%"
              />
              <el-select v-else-if="f.type === 'select'" v-model="formData[f.name]" style="width: 100%">
                <el-option v-for="o in f.options" :key="o" :label="o" :value="o" />
              </el-select>
              <el-input v-else v-model="formData[f.name]" />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :loading="submitting" @click="submitJob">
                提交求解作业
              </el-button>
              <el-button @click="runPreflight" :loading="preflightLoading">预检</el-button>
            </el-form-item>
          </el-form>

          <el-alert
            v-if="preflightResult"
            :title="preflightTitle"
            :type="preflightType"
            :closable="true"
            class="desc-alert"
          >
            <ul class="preflight-list">
              <li v-for="(c, i) in preflightChecks" :key="i">
                <strong>{{ c.name.replace(/_/g, ' ') }}</strong>：{{ c.message }}
              </li>
            </ul>
          </el-alert>
        </el-card>
      </el-col>

      <el-col :span="15">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>作业提交与监控</span>
              <el-button size="small" :icon="Refresh" @click="loadJobs">刷新作业列表</el-button>
            </div>
          </template>

          <!-- 当前提交的作业状态 -->
          <el-alert
            v-if="activeJob"
            :title="`作业 ${activeJob.job_id} — ${statusLabel(activeJob.status)}`"
            :type="statusAlertType(activeJob.status)"
            :closable="false"
            class="active-job-alert"
          >
            <template #default>
              <div>名称：{{ activeJob.name }}</div>
              <el-progress
                v-if="activeJob.status === 'running'"
                :percentage="progressPercent"
                :stroke-width="14"
                class="progress-bar"
              />
              <div v-if="activeJob.error" class="error-text">{{ activeJob.error }}</div>
            </template>
          </el-alert>

          <el-table :data="jobs" v-loading="jobsLoading" stripe size="small" style="width: 100%">
            <el-table-column prop="name" label="名称" min-width="180" show-overflow-tooltip />
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
            <el-table-column label="操作" width="110">
              <template #default="{ row }">
                <el-button
                  v-if="['queued', 'running'].includes(row.status)"
                  size="small"
                  type="warning"
                  @click="cancel(row)"
                >
                  取消
                </el-button>
                <el-button size="small" type="primary" link @click="goPostprocess(row)">
                  后处理
                </el-button>
              </template>
            </el-table-column>
            <template #empty><el-empty description="暂无作业" :image-size="60" /></template>
          </el-table>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Refresh } from '@element-plus/icons-vue'
import {
  cancelJob,
  getJob,
  getLiveMetrics,
  listJobs,
  preprocessPreflight,
  submitSolverJob,
  type Job,
} from '@/api/production'

// ---------------------------------------------------------------------------
// 求解器类型配置（字段与后端 Pydantic schema 对应）
// ---------------------------------------------------------------------------

interface SolverField {
  name: string
  label: string
  type: 'number' | 'select' | 'text'
  default?: any
  min?: number
  max?: number
  step?: number
  precision?: number
  options?: string[]
}

interface SolverTypeDef {
  label: string
  desc: string
  endpoint: string
  fields: SolverField[]
}

const solverTypes: Record<string, SolverTypeDef> = {
  cylinder_flow: {
    label: 'Cylinder Flow (2D)',
    desc: '二维圆柱绕流。对标 Williamson (1988) 的 Strouhal 数与阻力系数。',
    endpoint: '/solve/cylinder-flow',
    fields: [
      { name: 'nx', label: '网格宽 nx', type: 'number', default: 320, min: 20 },
      { name: 'ny', label: '网格高 ny', type: 'number', default: 100, min: 10 },
      { name: 'u_in', label: '入流速度', type: 'number', default: 0.08, step: 0.01, min: 0.001 },
      { name: 're', label: '雷诺数 Re', type: 'number', default: 100, min: 1 },
      { name: 'radius', label: '圆柱半径（格）', type: 'number', default: 12, min: 1 },
      { name: 'n_steps', label: '时间步数', type: 'number', default: 1200, min: 1 },
      { name: 'output_interval', label: '输出间隔', type: 'number', default: 200, min: 1 },
      { name: 'device', label: '设备', type: 'select', default: 'cpu', options: ['cpu', 'cuda:0'] },
      { name: 'seed', label: '随机种子', type: 'number', default: 0, min: 0 },
    ],
  },
  rotating_cylinder: {
    label: 'Rotating Cylinder (2D)',
    desc: '带自旋比的 Magnus 效应旋转圆柱，用于旋转体尾流与升力研究。',
    endpoint: '/solve/rotating-cylinder',
    fields: [
      { name: 'nx', label: '网格宽 nx', type: 'number', default: 320, min: 16 },
      { name: 'ny', label: '网格高 ny', type: 'number', default: 100, min: 8 },
      { name: 'u_in', label: '入流速度', type: 'number', default: 0.08, step: 0.01, min: 0.001 },
      { name: 're', label: '雷诺数 Re', type: 'number', default: 100, min: 1 },
      { name: 'radius', label: '圆柱半径', type: 'number', default: 12, min: 1 },
      { name: 'spin_ratio', label: '自旋比 α', type: 'number', default: 1.0, step: 0.1, min: 0 },
      { name: 'n_steps', label: '时间步数', type: 'number', default: 1200, min: 1 },
      { name: 'output_interval', label: '输出间隔', type: 'number', default: 200, min: 1 },
      { name: 'device', label: '设备', type: 'select', default: 'cpu', options: ['cpu', 'cuda:0'] },
      { name: 'seed', label: '随机种子', type: 'number', default: 0, min: 0 },
    ],
  },
  lid_driven_cavity: {
    label: 'Lid-Driven Cavity (2D)',
    desc: '顶盖驱动方腔流。对标 Ghia et al. (1982) 基准解。',
    endpoint: '/solve/lid-driven-cavity',
    fields: [
      { name: 'nx', label: '网格尺寸 nx (ny=nx)', type: 'number', default: 128, min: 8 },
      { name: 'u_lid', label: '顶盖速度', type: 'number', default: 0.1, step: 0.01, min: 0.001 },
      { name: 're', label: '雷诺数 Re', type: 'number', default: 100, min: 1 },
      { name: 'n_steps', label: '时间步数', type: 'number', default: 10000, min: 1 },
      { name: 'output_interval', label: '输出间隔', type: 'number', default: 2000, min: 1 },
      { name: 'device', label: '设备', type: 'select', default: 'cpu', options: ['cpu', 'cuda:0'] },
      { name: 'seed', label: '随机种子', type: 'number', default: 0, min: 0 },
    ],
  },
  backward_facing_step: {
    label: 'Backward-Facing Step (2D)',
    desc: '后向台阶流。测量再附着长度 x_r/h。',
    endpoint: '/solve/backward-facing-step',
    fields: [
      { name: 'nx', label: 'nx', type: 'number', default: 400, min: 20 },
      { name: 'ny', label: 'ny', type: 'number', default: 80, min: 6 },
      { name: 'step_h', label: '台阶高度（格）', type: 'number', default: 40, min: 1 },
      { name: 'x_step', label: '台阶前长度（格）', type: 'number', default: 80, min: 1 },
      { name: 'u_in', label: '入流速度', type: 'number', default: 0.05, step: 0.01 },
      { name: 're', label: '雷诺数 Re', type: 'number', default: 100, min: 1 },
      { name: 'n_steps', label: '时间步数', type: 'number', default: 30000, min: 1 },
      { name: 'output_interval', label: '输出间隔', type: 'number', default: 5000, min: 1 },
      { name: 'device', label: '设备', type: 'select', default: 'cpu', options: ['cpu', 'cuda:0'] },
    ],
  },
  turbulent_channel: {
    label: 'Turbulent Channel (2D LES)',
    desc: '体力驱动的 Smagorinsky LES 湍流通道流。验证对数律速度剖面。',
    endpoint: '/solve/turbulent-channel',
    fields: [
      { name: 'nx', label: 'nx', type: 'number', default: 256, min: 16 },
      { name: 'ny', label: 'ny', type: 'number', default: 64, min: 8 },
      { name: 're_tau', label: '摩擦雷诺数 Re_τ', type: 'number', default: 100, min: 1 },
      { name: 'u_tau', label: '摩擦速度 u_τ', type: 'number', default: 0.005, step: 0.001, min: 0.0001 },
      { name: 'smagorinsky_cs', label: 'Smagorinsky C_s', type: 'number', default: 0.1, step: 0.01 },
      { name: 'n_steps', label: '时间步数', type: 'number', default: 50000, min: 1 },
      { name: 'averaging_start', label: '统计起始步', type: 'number', default: 20000, min: 0 },
      { name: 'output_interval', label: '输出间隔', type: 'number', default: 5000, min: 1 },
      { name: 'device', label: '设备', type: 'select', default: 'cpu', options: ['cpu', 'cuda:0'] },
    ],
  },
  pipeline_flow: {
    label: 'Pipeline Flow (2D)',
    desc: '近底圆柱管道流（e/D 间隙比）。测量 Strouhal 数。',
    endpoint: '/solve/pipeline-flow',
    fields: [
      { name: 'nx', label: 'nx', type: 'number', default: 400, min: 20 },
      { name: 'ny', label: 'ny', type: 'number', default: 160, min: 10 },
      { name: 'diameter', label: '圆柱直径（格）', type: 'number', default: 20, min: 2 },
      { name: 'gap_ratio', label: '间隙比 e/D', type: 'number', default: 0.5, step: 0.1, min: 0 },
      { name: 'u_in', label: '入流速度', type: 'number', default: 0.05, step: 0.01 },
      { name: 're', label: '雷诺数 Re', type: 'number', default: 200, min: 1 },
      { name: 'n_steps', label: '时间步数', type: 'number', default: 30000, min: 1 },
      { name: 'output_interval', label: '输出间隔', type: 'number', default: 5000, min: 1 },
      { name: 'device', label: '设备', type: 'select', default: 'cpu', options: ['cpu', 'cuda:0'] },
    ],
  },
  dam_break: {
    label: 'Dam Break (2D 多相)',
    desc: '溃坝流。对标 Martin & Moyce (1952) 的前锋位置。',
    endpoint: '/solve/dam-break',
    fields: [
      { name: 'nx', label: 'nx', type: 'number', default: 400, min: 20 },
      { name: 'ny', label: 'ny', type: 'number', default: 200, min: 10 },
      { name: 'dam_width', label: '坝宽（格）', type: 'number', default: 100, min: 1 },
      { name: 'model', label: '多相模型', type: 'select', default: 'cg', options: ['sc', 'scmp', 'cg', 'fe'] },
      { name: 'rho_heavy', label: '重相密度', type: 'number', default: 0.8, step: 0.1 },
      { name: 'rho_light', label: '轻相密度', type: 'number', default: 0.4, step: 0.1 },
      { name: 'tau', label: '弛豫时间 τ', type: 'number', default: 1.0, step: 0.1, min: 0.51 },
      { name: 'n_steps', label: '时间步数', type: 'number', default: 4000, min: 1 },
      { name: 'output_interval', label: '输出间隔', type: 'number', default: 400, min: 1 },
      { name: 'device', label: '设备', type: 'select', default: 'cpu', options: ['cpu', 'cuda:0'] },
    ],
  },
  sloshing_tank: {
    label: 'Sloshing Tank (2D 多相)',
    desc: '受迫晃荡液舱。对标 Faltinsen (1978) 晃荡频率模型。',
    endpoint: '/solve/sloshing-tank',
    fields: [
      { name: 'nx', label: 'nx', type: 'number', default: 200, min: 16 },
      { name: 'ny', label: 'ny', type: 'number', default: 160, min: 16 },
      { name: 'water_level', label: '液位（格）', type: 'number', default: 80, min: 1 },
      { name: 'rho_water', label: '水密度', type: 'number', default: 0.8, step: 0.1 },
      { name: 'rho_air', label: '空气密度', type: 'number', default: 0.4, step: 0.1 },
      { name: 'tau', label: '弛豫时间 τ', type: 'number', default: 1.0, step: 0.1, min: 0.51 },
      { name: 'n_steps', label: '时间步数', type: 'number', default: 6000, min: 1 },
      { name: 'output_interval', label: '输出间隔', type: 'number', default: 600, min: 1 },
      { name: 'device', label: '设备', type: 'select', default: 'cpu', options: ['cpu', 'cuda:0'] },
    ],
  },
  sphere_flow: {
    label: 'Sphere Flow (3D D3Q19)',
    desc: '三维绕球流。测量阻力系数。',
    endpoint: '/solve/sphere-flow',
    fields: [
      { name: 'nx', label: 'nx', type: 'number', default: 120, min: 20 },
      { name: 'ny', label: 'ny', type: 'number', default: 60, min: 10 },
      { name: 'nz', label: 'nz', type: 'number', default: 60, min: 10 },
      { name: 'u_in', label: '入流速度', type: 'number', default: 0.06, step: 0.01 },
      { name: 're', label: '雷诺数 Re', type: 'number', default: 50, min: 1 },
      { name: 'radius', label: '球半径（格）', type: 'number', default: 8, min: 1 },
      { name: 'n_steps', label: '时间步数', type: 'number', default: 500, min: 1 },
      { name: 'output_interval', label: '输出间隔', type: 'number', default: 100, min: 1 },
      { name: 'device', label: '设备', type: 'select', default: 'cpu', options: ['cpu', 'cuda:0'] },
    ],
  },
  ship_hull: {
    label: 'Ship Hull – Wigley (3D)',
    desc: '三维 Wigley 船体阻力计算。报告阻力系数 Cd。',
    endpoint: '/solve/ship-hull',
    fields: [
      { name: 'nx', label: 'nx', type: 'number', default: 160, min: 20 },
      { name: 'ny', label: 'ny', type: 'number', default: 60, min: 10 },
      { name: 'nz', label: 'nz', type: 'number', default: 40, min: 10 },
      { name: 'u_in', label: '入流速度', type: 'number', default: 0.05, step: 0.01 },
      { name: 're', label: '雷诺数 Re', type: 'number', default: 200, min: 1 },
      { name: 'hull_length', label: '船长（格）', type: 'number', default: 80, min: 10 },
      { name: 'hull_beam', label: '船宽（格）', type: 'number', default: 8, min: 1 },
      { name: 'hull_draft', label: '吃水（格）', type: 'number', default: 12, min: 1 },
      { name: 'smagorinsky_cs', label: 'Smagorinsky C_s', type: 'number', default: 0.1, step: 0.01 },
      { name: 'wave_amp', label: '波幅（0=无）', type: 'number', default: 0, step: 0.5, min: 0 },
      { name: 'n_steps', label: '时间步数', type: 'number', default: 2000, min: 1 },
      { name: 'output_interval', label: '输出间隔', type: 'number', default: 200, min: 1 },
      { name: 'device', label: '设备', type: 'select', default: 'cpu', options: ['cpu', 'cuda:0'] },
    ],
  },
  suboff_wall_function: {
    label: 'SUBOFF – 高雷诺壁函数 (3D)',
    desc: '真实潜艇雷诺数 SUBOFF（Re~1e6+），τ 解耦对数律壁函数，Ct 对标 AFF-8 <1%。',
    endpoint: '/solve/suboff-wall-function',
    fields: [
      { name: 're', label: '雷诺数 Re', type: 'number', default: 2000000, min: 1 },
      { name: 'hull_type', label: '船型', type: 'select', default: 'full', options: ['bare_hull', 'with_sail', 'full'] },
      { name: 'nx', label: 'nx', type: 'number', default: 320, min: 80 },
      { name: 'ny', label: 'ny', type: 'number', default: 128, min: 32 },
      { name: 'nz', label: 'nz', type: 'number', default: 128, min: 32 },
      { name: 'hull_length', label: '船长（格）', type: 'number', default: 128, min: 20 },
      { name: 'u_in', label: '入流速度', type: 'number', default: 0.06, step: 0.01 },
      { name: 'n_steps', label: '时间步数', type: 'number', default: 5000, min: 100 },
      { name: 'output_interval', label: '输出间隔', type: 'number', default: 500, min: 10 },
      { name: 'device', label: '设备', type: 'select', default: 'cpu', options: ['cpu', 'cuda:0'] },
    ],
  },
  porous_drainage: {
    label: 'Porous Drainage (2D)',
    desc: '随机多孔介质中的两相渗流。',
    endpoint: '/solve/porous-drainage',
    fields: [
      { name: 'nx', label: 'nx', type: 'number', default: 160, min: 20 },
      { name: 'ny', label: 'ny', type: 'number', default: 80, min: 10 },
      { name: 'medium', label: '介质类型', type: 'select', default: 'random_cylinders', options: ['random_cylinders', 'tube_array'] },
      { name: 'model', label: '多相模型', type: 'select', default: 'cg', options: ['sc', 'cg'] },
      { name: 'porosity', label: '孔隙率', type: 'number', default: 0.6, step: 0.05, min: 0.1, max: 0.95 },
      { name: 'n_steps', label: '时间步数', type: 'number', default: 5000, min: 1 },
      { name: 'output_interval', label: '输出间隔', type: 'number', default: 1000, min: 1 },
      { name: 'device', label: '设备', type: 'select', default: 'cpu', options: ['cpu', 'cuda:0'] },
      { name: 'seed', label: '随机种子', type: 'number', default: 0, min: 0 },
    ],
  },
}

// ---------------------------------------------------------------------------
// 状态与表单
// ---------------------------------------------------------------------------

const router = useRouter()
const selectedType = ref<string>('cylinder_flow')
const formData = reactive<Record<string, any>>({})

const currentDef = computed<SolverTypeDef>(() => solverTypes[selectedType.value])
const currentFields = computed<SolverField[]>(() => currentDef.value.fields)
const currentDesc = computed(() => currentDef.value.desc)

const submitting = ref(false)
const preflightLoading = ref(false)
const preflightResult = ref<Record<string, any> | null>(null)

const preflightChecks = computed(() => (preflightResult.value?.checks as any[]) || [])
const preflightTitle = computed(() => {
  const r = preflightResult.value
  if (!r) return ''
  const errors = (r.errors as any[]) || []
  const warnings = (r.warnings as any[]) || []
  if (errors.length) return `预检发现 ${errors.length} 个错误`
  if (warnings.length) return `预检通过（${warnings.length} 个警告）`
  return '预检通过'
})
const preflightType = computed<'success' | 'warning' | 'error' | 'info'>(() => {
  const r = preflightResult.value
  if (!r) return 'info'
  const errors = (r.errors as any[]) || []
  const warnings = (r.warnings as any[]) || []
  if (errors.length) return 'error'
  if (warnings.length) return 'warning'
  return 'success'
})

function onTypeChange() {
  // 用默认值重建表单
  for (const f of currentFields.value) {
    formData[f.name] = f.default
  }
  // 清理旧字段
  for (const k of Object.keys(formData)) {
    if (!currentFields.value.some((f) => f.name === k)) {
      delete formData[k]
    }
  }
}

// 作业监控
const jobs = ref<Job[]>([])
const jobsLoading = ref(false)
const activeJob = ref<Job | null>(null)
const activeJobId = ref('')
const lastStep = ref(0)
const totalSteps = ref(0)

let pollTimer: ReturnType<typeof setInterval> | null = null
let listTimer: ReturnType<typeof setInterval> | null = null

const progressPercent = computed(() => {
  if (!totalSteps.value) return 0
  return Math.min(100, Math.round((lastStep.value / totalSteps.value) * 100))
})

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

function buildPayload(): Record<string, any> {
  const payload: Record<string, any> = {}
  for (const f of currentFields.value) {
    const v = formData[f.name]
    if (v !== undefined && v !== null && v !== '') {
      payload[f.name] = f.type === 'number' ? Number(v) : v
    }
  }
  return payload
}

async function submitJob() {
  submitting.value = true
  try {
    const payload = buildPayload()
    const res = await submitSolverJob(currentDef.value.endpoint, payload)
    activeJobId.value = res.job_id
    lastStep.value = 0
    totalSteps.value = Number(payload.n_steps) || 0
    ElMessage.success(`作业已提交：${res.job_id}`)
    await refreshActiveJob()
    startPolling()
    await loadJobs()
  } finally {
    submitting.value = false
  }
}

async function runPreflight() {
  preflightLoading.value = true
  try {
    const payload: Record<string, any> = { solver_type: selectedType.value }
    for (const f of currentFields.value) {
      const v = formData[f.name]
      if (v !== undefined && v !== null && v !== '') {
        payload[f.name] = f.type === 'number' ? Number(v) : v
      }
    }
    preflightResult.value = await preprocessPreflight(payload)
  } finally {
    preflightLoading.value = false
  }
}

async function refreshActiveJob() {
  if (!activeJobId.value) return
  try {
    const job = await getJob(activeJobId.value)
    activeJob.value = job
    totalSteps.value = Number((job.config?.n_steps as any) || totalSteps.value)
    // 拉取实时诊断数据（力系数/残差）用于进度
    try {
      const metrics = await getLiveMetrics(activeJobId.value, lastStep.value)
      if (metrics.diagnostics.length) {
        const last = metrics.diagnostics[metrics.diagnostics.length - 1]
        const step = Number(last.step ?? last.t ?? last.iter ?? 0)
        if (step > lastStep.value) lastStep.value = step
      }
    } catch {
      // 某些作业没有诊断数据，忽略
    }
    if (['completed', 'failed', 'cancelled'].includes(job.status)) {
      stopPolling()
    }
  } catch {
    // 作业查询失败时停止轮询
    stopPolling()
  }
}

function startPolling() {
  stopPolling()
  pollTimer = setInterval(() => {
    refreshActiveJob()
  }, 2000)
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

async function loadJobs() {
  jobsLoading.value = true
  try {
    const res = await listJobs({ limit: 0 })
    jobs.value = res.jobs
  } finally {
    jobsLoading.value = false
  }
}

async function cancel(row: Job) {
  try {
    await ElMessageBox.confirm(`确认取消作业「${row.name}」？`, '提示', { type: 'warning' })
  } catch {
    return
  }
  await cancelJob(row.job_id)
  ElMessage.success('已发送取消请求')
  await loadJobs()
}

function goPostprocess(row: Job) {
  router.push({ path: '/production/postprocess', query: { job_id: row.job_id } })
}

onMounted(() => {
  onTypeChange()
  loadJobs()
  listTimer = setInterval(() => {
    if (!activeJobId.value) loadJobs()
  }, 8000)
})

onBeforeUnmount(() => {
  stopPolling()
  if (listTimer) {
    clearInterval(listTimer)
    listTimer = null
  }
})
</script>

<style scoped>
.desc-alert {
  margin-bottom: 12px;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.active-job-alert {
  margin-bottom: 12px;
}
.progress-bar {
  margin-top: 8px;
}
.error-text {
  margin-top: 8px;
  color: #f56c6c;
  white-space: pre-wrap;
  font-family: monospace;
  font-size: 12px;
}
.preflight-list {
  margin: 0;
  padding-left: 16px;
}
.muted {
  color: #909399;
  font-size: 13px;
}
</style>
