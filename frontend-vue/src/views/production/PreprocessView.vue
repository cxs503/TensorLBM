<template>
  <div>
    <el-card>
      <el-tabs v-model="activeTab">
        <!-- 多边形掩码 -->
        <el-tab-pane :label="t('production.preprocess.tabPolygon')" name="polygon">
          <el-form label-width="110px" size="small" class="tab-form">
            <el-row :gutter="12">
              <el-col :span="8">
                <el-form-item :label="t('production.fields.meshWideNx')">
                  <el-input-number v-model="poly.nx" :min="10" controls-position="right" style="width: 100%" />
                </el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item :label="t('production.fields.meshHighNy')">
                  <el-input-number v-model="poly.ny" :min="10" controls-position="right" style="width: 100%" />
                </el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label=" ">
                  <el-button type="primary" :loading="polyLoading" @click="runPolygon">{{ t('production.preprocess.generateMask') }}</el-button>
                </el-form-item>
              </el-col>
            </el-row>
            <el-form-item :label="t('production.preprocess.vertices')">
              <el-input
                v-model="poly.verticesText"
                type="textarea"
                :rows="4"
                :placeholder="t('production.preprocess.verticesPlaceholder')"
              />
            </el-form-item>
          </el-form>
          <div v-if="polyResult" class="result-block">
            <img :src="polyResult.image" alt="mask" class="result-img" />
            <el-descriptions :column="2" size="small" border class="result-stats">
              <el-descriptions-item :label="t('production.preprocess.obstacleCells')">{{ polyResult.obstacle_cells }}</el-descriptions-item>
              <el-descriptions-item :label="t('production.common.fluidCells')">{{ polyResult.fluid_cells }}</el-descriptions-item>
              <el-descriptions-item :label="t('production.common.mesh')">{{ polyResult.nx }}×{{ polyResult.ny }}</el-descriptions-item>
            </el-descriptions>
          </div>
        </el-tab-pane>

        <!-- 随机孔隙率 -->
        <el-tab-pane :label="t('production.preprocess.tabPorosity')" name="porosity">
          <el-form label-width="120px" size="small" class="tab-form">
            <el-row :gutter="12">
              <el-col :span="6">
                <el-form-item label="nx"><el-input-number v-model="poro.nx" :min="16" controls-position="right" style="width: 100%" /></el-form-item>
              </el-col>
              <el-col :span="6">
                <el-form-item label="ny"><el-input-number v-model="poro.ny" :min="16" controls-position="right" style="width: 100%" /></el-form-item>
              </el-col>
              <el-col :span="6">
                <el-form-item :label="t('production.common.porosity')">
                  <el-input-number v-model="poro.porosity" :min="0.01" :max="0.99" :step="0.05" controls-position="right" style="width: 100%" />
                </el-form-item>
              </el-col>
              <el-col :span="6">
                <el-form-item :label="t('production.common.seed')">
                  <el-input-number v-model="poro.seed" :min="0" controls-position="right" style="width: 100%" />
                </el-form-item>
              </el-col>
            </el-row>
            <el-button type="primary" :loading="poroLoading" @click="runPorosity">{{ t('production.preprocess.generatePorosityMask') }}</el-button>
          </el-form>
          <div v-if="poroResult" class="result-block">
            <img :src="poroResult.image" alt="porosity" class="result-img" />
            <el-descriptions :column="2" size="small" border class="result-stats">
              <el-descriptions-item :label="t('production.preprocess.targetPorosity')">{{ poroResult.requested_porosity }}</el-descriptions-item>
              <el-descriptions-item :label="t('production.preprocess.actualPorosity')">{{ poroResult.actual_porosity }}</el-descriptions-item>
            </el-descriptions>
          </div>
        </el-tab-pane>

        <!-- 单位换算 -->
        <el-tab-pane :label="t('production.preprocess.tabUnits')" name="units">
          <el-form label-width="140px" size="small" class="tab-form">
            <el-row :gutter="12">
              <el-col :span="8">
                <el-form-item :label="t('production.preprocess.charLengthM')"><el-input-number v-model="units.phys_length_m" :min="0.001" controls-position="right" style="width: 100%" /></el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item :label="t('production.preprocess.charVelocityMs')"><el-input-number v-model="units.phys_velocity_ms" :min="0.001" controls-position="right" style="width: 100%" /></el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item :label="t('production.common.nuM2s')">
                  <el-input-number v-model="units.phys_nu_m2s" :min="1e-9" :step="1e-7" :precision="9" controls-position="right" style="width: 100%" />
                </el-form-item>
              </el-col>
            </el-row>
            <el-row :gutter="12">
              <el-col :span="8">
                <el-form-item :label="t('production.preprocess.lbmLength')"><el-input-number v-model="units.lbm_length" :min="10" controls-position="right" style="width: 100%" /></el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item :label="t('production.preprocess.lbmVelocity')"><el-input-number v-model="units.lbm_velocity" :min="0.01" :step="0.01" controls-position="right" style="width: 100%" /></el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label=" ">
                  <el-button type="primary" :loading="unitsLoading" @click="runUnits">{{ t('production.preprocess.convert') }}</el-button>
                </el-form-item>
              </el-col>
            </el-row>
          </el-form>
          <el-table v-if="unitsResult" :data="unitsRows" size="small" border class="result-stats">
            <el-table-column prop="label" :label="t('production.common.param')" width="180" />
            <el-table-column prop="value" :label="t('production.common.value')" />
          </el-table>
        </el-tab-pane>

        <!-- Y+ -->
        <el-tab-pane :label="t('production.preprocess.tabYplus')" name="yplus">
          <el-form label-width="140px" size="small" class="tab-form">
            <el-row :gutter="12">
              <el-col :span="8">
                <el-form-item :label="t('production.common.reynoldsRe')"><el-input-number v-model="yp.re" :min="1" controls-position="right" style="width: 100%" /></el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item :label="t('production.preprocess.freeStreamVelocityMs')"><el-input-number v-model="yp.u_ms" :min="0.001" controls-position="right" style="width: 100%" /></el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item :label="t('production.preprocess.charLengthM')"><el-input-number v-model="yp.l_m" :min="0.001" controls-position="right" style="width: 100%" /></el-form-item>
              </el-col>
            </el-row>
            <el-row :gutter="12">
              <el-col :span="8">
                <el-form-item :label="t('production.common.nuM2s')">
                  <el-input-number v-model="yp.nu_m2s" :min="1e-9" :step="1e-6" :precision="8" controls-position="right" style="width: 100%" />
                </el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item :label="t('production.preprocess.targetYplus')"><el-input-number v-model="yp.target_yplus" :min="0.01" controls-position="right" style="width: 100%" /></el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item :label="t('production.preprocess.cellsAlongL')"><el-input-number v-model="yp.n_cells" :min="1" controls-position="right" style="width: 100%" /></el-form-item>
              </el-col>
            </el-row>
            <el-row :gutter="12">
              <el-col :span="8">
                <el-form-item :label="t('production.preprocess.geometryType')">
                  <el-select v-model="yp.geometry" style="width: 100%">
                    <el-option :label="t('production.preprocess.geoFlatPlate')" value="flat_plate" />
                    <el-option :label="t('production.preprocess.geoCylinder')" value="cylinder" />
                    <el-option :label="t('production.preprocess.geoChannel')" value="channel" />
                  </el-select>
                </el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label=" ">
                  <el-button type="primary" :loading="ypLoading" @click="runYPlus">{{ t('production.preprocess.calculate') }}</el-button>
                </el-form-item>
              </el-col>
            </el-row>
          </el-form>
          <el-table v-if="ypResult" :data="ypRows" size="small" border class="result-stats">
            <el-table-column prop="label" :label="t('production.common.param')" width="220" />
            <el-table-column prop="value" :label="t('production.common.value')" />
          </el-table>
          <el-alert v-if="ypNote" :title="ypNote" type="info" :closable="false" class="result-stats" />
        </el-tab-pane>

        <!-- 流体材料库 -->
        <el-tab-pane :label="t('production.preprocess.tabMaterials')" name="materials">
          <div class="tab-form">
            <el-button size="small" :loading="materialsLoading" @click="loadMaterials">{{ t('production.preprocess.loadMaterials') }}</el-button>
            <el-radio-group v-model="materialCategory" size="small" style="margin-left: 12px" @change="loadMaterials">
              <el-radio-button label="">{{ t('production.preprocess.all') }}</el-radio-button>
              <el-radio-button label="liquid">{{ t('production.preprocess.liquid') }}</el-radio-button>
              <el-radio-button label="gas">{{ t('production.preprocess.gas') }}</el-radio-button>
            </el-radio-group>
          </div>
          <el-table :data="materials" v-loading="materialsLoading" size="small" border class="result-stats">
            <el-table-column prop="name" :label="t('production.common.name')" min-width="200" />
            <el-table-column prop="name_zh" :label="t('production.preprocess.nameZh')" width="140" />
            <el-table-column prop="category" :label="t('production.preprocess.category')" width="80" />
            <el-table-column :label="t('production.preprocess.density')" width="120">
              <template #default="{ row }">{{ row.density_kg_m3 }}</template>
            </el-table-column>
            <el-table-column :label="t('production.common.nuM2s')" width="150">
              <template #default="{ row }">{{ row.kinematic_viscosity_m2_s.toExponential(3) }}</template>
            </el-table-column>
            <el-table-column :label="t('production.preprocess.refTemp')" width="110">
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
import { useI18n } from 'vue-i18n'
import {
  preprocessMaterials,
  preprocessPolygonMask,
  preprocessRandomPorosity,
  preprocessUnits,
  preprocessYPlus,
  type Material,
} from '@/api/production'

const { t } = useI18n()

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
    { label: t('production.common.reynoldsRe'), value: String(r.reynolds_number) },
    { label: t('production.preprocess.lbmNu'), value: String(r.lbm_nu) },
    { label: t('production.preprocess.lbmTau'), value: `${r.lbm_tau} ${r.stable ? t('production.common.stable') : t('production.common.unstable')}` },
    { label: 'dx (m)', value: String(r.dx_m) },
    { label: 'dt (s)', value: String(r.dt_s) },
    { label: t('production.preprocess.machMa'), value: String(r.mach_number) },
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
    { label: t('production.preprocess.cf'), value: String(r.c_f) },
    { label: t('production.preprocess.uTau'), value: String(r.u_tau_ms) },
    { label: t('production.preprocess.deltaY'), value: String(r.delta_y_m) },
    { label: t('production.preprocess.deltaYLbm'), value: String(r.delta_y_lbm) },
    { label: t('production.preprocess.dx'), value: String(r.dx_m) },
    { label: t('production.preprocess.blThickness'), value: String(r.bl_thickness_m) },
    { label: t('production.preprocess.cellsInsideBl'), value: String(r.cells_inside_bl) },
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
