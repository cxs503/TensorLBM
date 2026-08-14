<template>
  <div>
    <el-card>
      <el-tabs v-model="activeTab">
        <!-- 多边形掩码 -->
        <el-tab-pane label="多边形 → 2D 掩码" name="polygon">
          <el-form label-width="110px" size="small" class="tab-form">
            <el-row :gutter="12">
              <el-col :span="8">
                <el-form-item label="网格宽 nx">
                  <el-input-number v-model="poly.nx" :min="10" controls-position="right" style="width: 100%" />
                </el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="网格高 ny">
                  <el-input-number v-model="poly.ny" :min="10" controls-position="right" style="width: 100%" />
                </el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label=" ">
                  <el-button type="primary" :loading="polyLoading" @click="runPolygon">生成掩码</el-button>
                </el-form-item>
              </el-col>
            </el-row>
            <el-form-item label="顶点坐标">
              <el-input
                v-model="poly.verticesText"
                type="textarea"
                :rows="4"
                placeholder="每行一个顶点，格式：x,y （像素坐标）"
              />
            </el-form-item>
          </el-form>
          <div v-if="polyResult" class="result-block">
            <img :src="polyResult.image" alt="mask" class="result-img" />
            <el-descriptions :column="2" size="small" border class="result-stats">
              <el-descriptions-item label="障碍网格">{{ polyResult.obstacle_cells }}</el-descriptions-item>
              <el-descriptions-item label="流体网格">{{ polyResult.fluid_cells }}</el-descriptions-item>
              <el-descriptions-item label="网格">{{ polyResult.nx }}×{{ polyResult.ny }}</el-descriptions-item>
            </el-descriptions>
          </div>
        </el-tab-pane>

        <!-- 随机孔隙率 -->
        <el-tab-pane label="随机孔隙率掩码" name="porosity">
          <el-form label-width="120px" size="small" class="tab-form">
            <el-row :gutter="12">
              <el-col :span="6">
                <el-form-item label="nx"><el-input-number v-model="poro.nx" :min="16" controls-position="right" style="width: 100%" /></el-form-item>
              </el-col>
              <el-col :span="6">
                <el-form-item label="ny"><el-input-number v-model="poro.ny" :min="16" controls-position="right" style="width: 100%" /></el-form-item>
              </el-col>
              <el-col :span="6">
                <el-form-item label="孔隙率">
                  <el-input-number v-model="poro.porosity" :min="0.01" :max="0.99" :step="0.05" controls-position="right" style="width: 100%" />
                </el-form-item>
              </el-col>
              <el-col :span="6">
                <el-form-item label="随机种子">
                  <el-input-number v-model="poro.seed" :min="0" controls-position="right" style="width: 100%" />
                </el-form-item>
              </el-col>
            </el-row>
            <el-button type="primary" :loading="poroLoading" @click="runPorosity">生成孔隙率掩码</el-button>
          </el-form>
          <div v-if="poroResult" class="result-block">
            <img :src="poroResult.image" alt="porosity" class="result-img" />
            <el-descriptions :column="2" size="small" border class="result-stats">
              <el-descriptions-item label="目标孔隙率">{{ poroResult.requested_porosity }}</el-descriptions-item>
              <el-descriptions-item label="实际孔隙率">{{ poroResult.actual_porosity }}</el-descriptions-item>
            </el-descriptions>
          </div>
        </el-tab-pane>

        <!-- 单位换算 -->
        <el-tab-pane label="物理单位 → LBM 换算" name="units">
          <el-form label-width="140px" size="small" class="tab-form">
            <el-row :gutter="12">
              <el-col :span="8">
                <el-form-item label="特征长度 (m)"><el-input-number v-model="units.phys_length_m" :min="0.001" controls-position="right" style="width: 100%" /></el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="特征速度 (m/s)"><el-input-number v-model="units.phys_velocity_ms" :min="0.001" controls-position="right" style="width: 100%" /></el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="运动粘度 (m²/s)">
                  <el-input-number v-model="units.phys_nu_m2s" :min="1e-9" :step="1e-7" :precision="9" controls-position="right" style="width: 100%" />
                </el-form-item>
              </el-col>
            </el-row>
            <el-row :gutter="12">
              <el-col :span="8">
                <el-form-item label="LBM 长度"><el-input-number v-model="units.lbm_length" :min="10" controls-position="right" style="width: 100%" /></el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="LBM 速度"><el-input-number v-model="units.lbm_velocity" :min="0.01" :step="0.01" controls-position="right" style="width: 100%" /></el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label=" ">
                  <el-button type="primary" :loading="unitsLoading" @click="runUnits">换算</el-button>
                </el-form-item>
              </el-col>
            </el-row>
          </el-form>
          <el-table v-if="unitsResult" :data="unitsRows" size="small" border class="result-stats">
            <el-table-column prop="label" label="参数" width="180" />
            <el-table-column prop="value" label="数值" />
          </el-table>
        </el-tab-pane>

        <!-- Y+ -->
        <el-tab-pane label="Y+ 首层网格高度" name="yplus">
          <el-form label-width="140px" size="small" class="tab-form">
            <el-row :gutter="12">
              <el-col :span="8">
                <el-form-item label="雷诺数 Re"><el-input-number v-model="yp.re" :min="1" controls-position="right" style="width: 100%" /></el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="来流速度 (m/s)"><el-input-number v-model="yp.u_ms" :min="0.001" controls-position="right" style="width: 100%" /></el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="特征长度 (m)"><el-input-number v-model="yp.l_m" :min="0.001" controls-position="right" style="width: 100%" /></el-form-item>
              </el-col>
            </el-row>
            <el-row :gutter="12">
              <el-col :span="8">
                <el-form-item label="运动粘度 (m²/s)">
                  <el-input-number v-model="yp.nu_m2s" :min="1e-9" :step="1e-6" :precision="8" controls-position="right" style="width: 100%" />
                </el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="目标 y+"><el-input-number v-model="yp.target_yplus" :min="0.01" controls-position="right" style="width: 100%" /></el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="沿 L 网格数"><el-input-number v-model="yp.n_cells" :min="1" controls-position="right" style="width: 100%" /></el-form-item>
              </el-col>
            </el-row>
            <el-row :gutter="12">
              <el-col :span="8">
                <el-form-item label="几何类型">
                  <el-select v-model="yp.geometry" style="width: 100%">
                    <el-option label="平板" value="flat_plate" />
                    <el-option label="圆柱" value="cylinder" />
                    <el-option label="管道" value="channel" />
                  </el-select>
                </el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label=" ">
                  <el-button type="primary" :loading="ypLoading" @click="runYPlus">计算</el-button>
                </el-form-item>
              </el-col>
            </el-row>
          </el-form>
          <el-table v-if="ypResult" :data="ypRows" size="small" border class="result-stats">
            <el-table-column prop="label" label="参数" width="220" />
            <el-table-column prop="value" label="数值" />
          </el-table>
          <el-alert v-if="ypNote" :title="ypNote" type="info" :closable="false" class="result-stats" />
        </el-tab-pane>

        <!-- 流体材料库 -->
        <el-tab-pane label="流体材料库" name="materials">
          <div class="tab-form">
            <el-button size="small" :loading="materialsLoading" @click="loadMaterials">加载材料库</el-button>
            <el-radio-group v-model="materialCategory" size="small" style="margin-left: 12px" @change="loadMaterials">
              <el-radio-button label="">全部</el-radio-button>
              <el-radio-button label="liquid">液体</el-radio-button>
              <el-radio-button label="gas">气体</el-radio-button>
            </el-radio-group>
          </div>
          <el-table :data="materials" v-loading="materialsLoading" size="small" border class="result-stats">
            <el-table-column prop="name" label="名称" min-width="200" />
            <el-table-column prop="name_zh" label="中文名" width="140" />
            <el-table-column prop="category" label="类别" width="80" />
            <el-table-column label="密度 (kg/m³)" width="120">
              <template #default="{ row }">{{ row.density_kg_m3 }}</template>
            </el-table-column>
            <el-table-column label="运动粘度 (m²/s)" width="150">
              <template #default="{ row }">{{ row.kinematic_viscosity_m2_s.toExponential(3) }}</template>
            </el-table-column>
            <el-table-column label="参考温度 (°C)" width="110">
              <template #default="{ row }">{{ row.ref_temp_c }}</template>
            </el-table-column>
          </el-table>
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import {
  preprocessMaterials,
  preprocessPolygonMask,
  preprocessRandomPorosity,
  preprocessUnits,
  preprocessYPlus,
  type Material,
} from '@/api/production'

const activeTab = ref('polygon')

// 多边形掩码
const poly = reactive({
  nx: 200,
  ny: 100,
  verticesText: '50,20\n150,20\n170,80\n30,80',
})
const polyLoading = ref(false)
const polyResult = ref<Record<string, any> | null>(null)

async function runPolygon() {
  const vertices = poly.verticesText
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => l.split(',').map(Number))
  if (vertices.length < 3) {
    return
  }
  polyLoading.value = true
  try {
    polyResult.value = await preprocessPolygonMask({ nx: poly.nx, ny: poly.ny, vertices })
  } finally {
    polyLoading.value = false
  }
}

// 随机孔隙率
const poro = reactive({ nx: 128, ny: 128, porosity: 0.4, sigma: 0, seed: 0 })
const poroLoading = ref(false)
const poroResult = ref<Record<string, any> | null>(null)

async function runPorosity() {
  poroLoading.value = true
  try {
    poroResult.value = await preprocessRandomPorosity({ ...poro })
  } finally {
    poroLoading.value = false
  }
}

// 单位换算
const units = reactive({
  phys_length_m: 1,
  phys_velocity_ms: 1,
  phys_nu_m2s: 1e-6,
  lbm_length: 100,
  lbm_velocity: 0.1,
})
const unitsLoading = ref(false)
const unitsResult = ref<Record<string, any> | null>(null)

const unitsRows = computed(() => {
  const r = unitsResult.value
  if (!r) return []
  return [
    { label: '雷诺数 Re', value: String(r.reynolds_number) },
    { label: 'LBM 运动粘度 ν', value: String(r.lbm_nu) },
    { label: 'LBM 弛豫时间 τ', value: `${r.lbm_tau} ${r.stable ? '（稳定）' : '（不稳定）'}` },
    { label: 'dx (m)', value: String(r.dx_m) },
    { label: 'dt (s)', value: String(r.dt_s) },
    { label: '马赫数 Ma', value: String(r.mach_number) },
  ]
})

async function runUnits() {
  unitsLoading.value = true
  try {
    unitsResult.value = await preprocessUnits({ ...units })
  } finally {
    unitsLoading.value = false
  }
}

// Y+
const yp = reactive({
  re: 1e5,
  u_ms: 1,
  l_m: 1,
  nu_m2s: 1e-5,
  target_yplus: 1,
  n_cells: 100,
  geometry: 'flat_plate',
})
const ypLoading = ref(false)
const ypResult = ref<Record<string, any> | null>(null)

const ypRows = computed(() => {
  const r = ypResult.value
  if (!r) return []
  return [
    { label: '摩擦系数 Cf', value: String(r.c_f) },
    { label: '摩擦速度 u_τ (m/s)', value: String(r.u_tau_ms) },
    { label: '首层网格高度 Δy (m)', value: String(r.delta_y_m) },
    { label: 'LBM 首层高度 Δy_LBM', value: String(r.delta_y_lbm) },
    { label: '网格尺寸 dx (m)', value: String(r.dx_m) },
    { label: '边界层厚度 (m)', value: String(r.bl_thickness_m) },
    { label: '边界层内网格数', value: String(r.cells_inside_bl) },
  ]
})
const ypNote = computed(() => (ypResult.value?.note as string) || '')

async function runYPlus() {
  ypLoading.value = true
  try {
    ypResult.value = await preprocessYPlus({ ...yp })
  } finally {
    ypLoading.value = false
  }
}

// 材料库
const materials = ref<Material[]>([])
const materialsLoading = ref(false)
const materialCategory = ref('')

async function loadMaterials() {
  materialsLoading.value = true
  try {
    const res = await preprocessMaterials(materialCategory.value || undefined)
    materials.value = res.materials
  } finally {
    materialsLoading.value = false
  }
}

onMounted(() => {
  loadMaterials()
})
</script>

<style scoped>
.tab-form {
  margin-bottom: 8px;
}
.result-block {
  display: flex;
  flex-direction: column;
  align-items: center;
}
.result-img {
  max-width: 100%;
  max-height: 360px;
  margin-bottom: 12px;
}
.result-stats {
  width: 100%;
  margin-top: 12px;
}
</style>
