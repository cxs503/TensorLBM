import subprocess, os
PGENV = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
def q(sql):
    r = subprocess.run(["runuser","-u","postgres","--","psql","-d","postgres","-tA","-F","|","-c",sql],
                       capture_output=True, text=True, env=PGENV)
    if r.returncode != 0:
        return "ERR:" + r.stderr.strip()[:200]
    return r.stdout

print("DBLIST:", q("select datname from pg_database;"))