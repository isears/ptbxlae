from simple_slurm import Slurm
import argparse
import subprocess
import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Low-overhead script to run on login node and submit tuning jobs as necessary"
    )

    parser.add_argument("script_path", type=str, help="Path to python tuning script")
    parser.add_argument(
        "--max_jobs",
        type=int,
        default=8,
        help="Maximum number of jobs to have running at one time",
    )
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def refresh_active_joblist(job_ids: list[int]):
    result = subprocess.run(
        ["squeue", "-j", ",".join([str(id) for id in job_ids]), "-h"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Error running squeue: {result.stderr.strip()}")

    result_raw = result.stdout.strip()

    active_jobs = [int(r.split()[0]) for r in result_raw.split("\n") if len(r) > 0]

    return [j for j in job_ids if j in active_jobs]


if __name__ == "__main__":
    args = parse_args()

    joblist = list()
    total_jobcount = 0
    job_timelimit = datetime.timedelta(hours=12)

    while True:
        try:
            joblist = refresh_active_joblist(joblist)

            if len(joblist) < args.max_jobs:

                if args.debug:
                    job_timelimit = datetime.timedelta(seconds=30)
                    s = Slurm(
                        job_name=f"tuning-{total_jobcount}",
                        N=1,
                        time=str(job_timelimit + datetime.timedelta(minutes=5)),
                        o="slurmlogs/tunejob.%j.out",
                        p="debug",
                    )

                else:
                    s = Slurm(
                        job_name=f"tuning-{total_jobcount}",
                        N=1,
                        time=str(job_timelimit + datetime.timedelta(minutes=5)),
                        o="slurmlogs/tunejob.%j.out",
                        p="gpu",
                        gres="gpu:1",
                        c=8,
                        mem="32G",
                    )

                jobid = s.sbatch(
                    f"python {args.script_path} --timelimit {job_timelimit.total_seconds() / 60}",
                    verbose=False,
                )
                joblist.append(jobid)

                print(f"Starting job: {jobid} with timelimit {str(job_timelimit)}")
                total_jobcount += 1
        except KeyboardInterrupt:
            print("Shutting down...")
            print(f"Launched {total_jobcount} jobs")
            print("Remaining active jobs:")
            for j in joblist:
                print(j)

            quit()
