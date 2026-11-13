import urllib.request, json, base64, sys
BASE='http://127.0.0.1:8010/api/v1'
def _tok():
    body=json.dumps({'username':'eco_admin','password':'EcoTest!2026pass'}).encode()
    req=urllib.request.Request(BASE+'/auth/login',data=body,headers={'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(req))['access_token']
TOK=_tok()
def call(path, method='GET', body=None):
    req=urllib.request.Request(BASE+path, method=method,
        headers={'Authorization':'Bearer '+TOK,'Content-Type':'application/json'})
    data=json.dumps(body).encode() if body is not None else None
    try:
        r=urllib.request.urlopen(req, data=data)
        return json.load(r)
    except Exception as e:
        try: return {'_err':str(e), 'body': e.read().decode()[:800]}
        except: return {'_err':str(e)}
if __name__=='__main__':
    print(json.dumps(call(sys.argv[1]), ensure_ascii=False)[:int(sys.argv[2]) if len(sys.argv)>2 else 3000])
