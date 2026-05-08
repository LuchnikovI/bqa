#!/usr/bin/env bash

set -euo pipefail

(echo '{"degree" : 4, "nodes_number" : 200}' | random_regular_graph | metrics) &> /dev/null &&
(echo '{"degree" : 4, "nodes_number" : 200}' | random_regular_graph | validate | metrics) &> /dev/null &&
(echo '{"degree" : 4, "nodes_number" : 200}' | random_regular_graph | validate | canonicalize | metrics) &> /dev/null &&
(echo '{"degree" : 4, "nodes_number" : 200}' | random_regular_graph | canonicalize | canonicalize | metrics) &> /dev/null &&
(echo '{"degree" : 4, "nodes_number" : 200}' | random_regular_graph | validate | full_preprocess | metrics) &> /dev/null &&
(echo '{"degree" : 4, "nodes_number" : 200}' | random_regular_graph | full_preprocess | full_preprocess | metrics) &> /dev/null &&
(echo '{"degree" : 4, "nodes_number" : 200}' | random_regular_graph | full_preprocess | canonicalize | validate | metrics) &> /dev/null &&
(echo '{"degree" : 4, "nodes_number" : 200}' | random_regular_graph | sparsify | metrics) &> /dev/null &&
(echo '{"degree" : 4, "nodes_number" : 200}' | random_regular_graph | sparsify | validate | metrics) &> /dev/null &&
(echo '{"degree" : 4, "nodes_number" : 200}' | random_regular_graph | sparsify | validate | canonicalize | metrics) &> /dev/null &&
(echo '{"degree" : 4, "nodes_number" : 200}' | random_regular_graph | sparsify | canonicalize | canonicalize | metrics) &> /dev/null &&
(echo '{"degree" : 4, "nodes_number" : 200}' | random_regular_graph | sparsify | validate | full_preprocess | metrics) &> /dev/null &&
(echo '{"degree" : 4, "nodes_number" : 200}' | random_regular_graph | sparsify | full_preprocess | full_preprocess | metrics) &> /dev/null &&
(echo '{"degree" : 4, "nodes_number" : 200}' | random_regular_graph | sparsify | full_preprocess | canonicalize | validate | metrics) &> /dev/null &&
(echo '{"degree" : 4, "nodes_number" : 200}' | random_regular_graph | adjust_schedule | sparsify | metrics) &> /dev/null &&
(echo '{"degree" : 4, "nodes_number" : 200}' | random_regular_graph | adjust_schedule | sparsify | validate | metrics) &> /dev/null &&
(echo '{"degree" : 4, "nodes_number" : 200}' | random_regular_graph | adjust_schedule | sparsify | validate | canonicalize | metrics) &> /dev/null &&
(echo '{"degree" : 4, "nodes_number" : 200}' | random_regular_graph | adjust_schedule | sparsify | canonicalize | canonicalize | metrics) &> /dev/null &&
(echo '{"degree" : 4, "nodes_number" : 200}' | random_regular_graph | adjust_schedule | sparsify | validate | full_preprocess | metrics) &> /dev/null &&
(echo '{"degree" : 4, "nodes_number" : 200}' | random_regular_graph | adjust_schedule | sparsify | adjust_schedule | full_preprocess | full_preprocess | metrics) &> /dev/null &&
(echo '{"degree" : 4, "nodes_number" : 200}' | random_regular_graph | sparsify | full_preprocess | adjust_schedule | canonicalize | validate | metrics) &> /dev/null

