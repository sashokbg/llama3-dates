unset OTEL_SDK_DISABLED
pf config set telemetry.enabled=true

pf run delete -n 'default_run' -y
pf run delete -n 'default_run_eval' -y
pf trace delete --run default_run -y

pf run create --flow default --data ./default/data.jsonl --column-mapping query='${data.query}' -n default_run

pf config set telemetry.enabled=false

export OTEL_SDK_DISABLED=true
pf run create --flow eval-accuracy \
  --data default/data.jsonl \
  --column-mapping ground_truth='${data.expected_answer}' \
  --column-mapping prediction='${run.outputs.date_result}' \
  --run default_run -n default_run_eval --stream

pf run show-metrics -n default_run_eval
pf run visualize --names 'default_run, default_run_eval'

