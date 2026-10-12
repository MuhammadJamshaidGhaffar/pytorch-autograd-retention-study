from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parents[2]
subprocess.run([sys.executable,str(root/"benchmark/measure.py"),"--case",Path(__file__).parent.name],check=True)
