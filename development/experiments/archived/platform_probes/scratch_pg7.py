import subprocess
PGENV = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
def q(sql, db="tensorplatform"):
    r = subprocess.run(["runuser","-u","postgres","--","psql","-d",db,"-tA","-F","|","-c",sql],
                       capture_output=True, text=True, env=PGENV)
    return r.stdout if r.returncode == 0 else "ERR:" + r.stderr.strip()[:300]

print("== node")
print(q("""select id||' | '||node_id||' | '||title||' ||| '||coalesce(goal,'')||' ||| ACC: '||coalesce(acceptance_criteria,'')
from campaign_plan_nodes where title ilike '%R4-COEXISTENCE%'"""))