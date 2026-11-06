#!/usr/bin/env bash
# Summarize testgpuerror log: current cfg vs Eigen reference (avg / 90% / 99% / max)
f=$1
printf "%-22s" "$(basename $f .log)"
for k in winrateError leadError topPolicyDelta policyKLDiv ownershipError; do
  v=$(grep "^: current cfg error vs reference $k" "$f" | awk '{print $(NF-3), $(NF-1)}'); printf " %s[%s]" "$k" "$v"
done
echo; grep "^: current cfg error vs reference closest margin" "$f" | sed 's/^: current cfg error vs reference closest margin: */    margin: /'
grep -i -E "fail|exceed|what\(\)" "$f" | head -3
