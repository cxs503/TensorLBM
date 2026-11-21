import axios from 'axios'
import type { AxiosInstance, AxiosRequestConfig } from 'axios'
import { ElMessage } from 'element-plus'

const instance: AxiosInstance = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

instance.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const msg = error.response?.data?.detail || error.message || '请求失败'
    ElMessage.error(msg)
    return Promise.reject(error)
  },
)

export function request<T = any>(config: AxiosRequestConfig): Promise<T> {
  return instance.request(config) as unknown as Promise<T>
}

export const api = {
  get: <T = any>(url: string, params?: object) =>
    request<T>({ method: 'get', url, params }),
  post: <T = any>(url: string, data?: object) =>
    request<T>({ method: 'post', url, data }),
  put: <T = any>(url: string, data?: object) =>
    request<T>({ method: 'put', url, data }),
  delete: <T = any>(url: string, params?: object) =>
    request<T>({ method: 'delete', url, params }),
}

export default instance
