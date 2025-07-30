from mlflow.tracking import MlflowClient

def delete_runs_except_n_latest(experiment_id, n_latest):
  client = MlflowClient()

  runs = client.search_runs(
    experiment_ids=[experiment_id], 
    order_by=["start_time DESC"],
    max_results=50000
  )

  runs_to_delete = runs[n_latest:]
  for run in runs_to_delete:
    client.delete_run(run.info.run_id)
    print(f"Deleted run: {run.info.run_id}")

experiment_id = "142554258142140382"
n_latest_to_keep = 35

delete_runs_except_n_latest(experiment_id, n_latest_to_keep)
