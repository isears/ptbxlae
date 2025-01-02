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


def check_none_failed(job_ids: list[int]):
    # sacct -n -X -j 7492626,7492628,7492643
    result = subprocess.run(
        ["sacct", "-n", "-X", "-j", ",".join([str(id) for id in job_ids])],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Error running sacct: {result.stderr.strip()}")

    result_raw = result.stdout.strip()

    for line in result_raw.split("\n"):
        if "FAILED" in line:
            raise RuntimeError("Detected Failed job")


if __name__ == "__main__":
    args = parse_args()

    active_joblist = list()
    all_joblist = list()
    total_jobcount = 0
    job_timelimit = datetime.timedelta(hours=12)

    while True:
        try:
            active_joblist = refresh_active_joblist(active_joblist)

            if len(active_joblist) < args.max_jobs:

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
                    f"""
                    source ~/.bashrc
                    conda activate ptbxlae
                    which python
                    pwd -P
                    python {args.script_path} --timelimit {job_timelimit.total_seconds() / 60}
                    """,
                    verbose=False,
                )
                active_joblist.append(jobid)
                all_joblist.append(jobid)

                print(f"Starting job: {jobid} with timelimit {str(job_timelimit)}")
                total_jobcount += 1

                # Once per iteration make sure we aren't just spawning failed jobs
                check_none_failed(all_joblist)
        except KeyboardInterrupt:
            print("Shutting down...")
            print(f"Launched {total_jobcount} jobs")
            print("Remaining active jobs:")
            for j in active_joblist:
                print(j)

            quit()
