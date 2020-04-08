#!/bin/bash

function lang_path {
  echo "../resources/$1.jsonl.gz"
}

m="../cache/models/2020.04.06_18.24.03-go-java-256/2.pickle"
c="../cache/models/2020.04.06_18.24.03-go-java-256/config.toml"
o="../cache/models/2020.04.06_18.24.03-go-java-256/2.csv"
python -m deepcode.predict \
  -q ../resources/queries.csv \
  -d "$(lang_path java)" "$(lang_path go)" \
  -o "$o" \
  -m "$m"\
  -c "$c"

m="../cache/models/2020.04.07_06.43.21-go-javascript-256/2.pickle"
c="../cache/models/2020.04.07_06.43.21-go-javascript-256/config.toml"
o="../cache/models/2020.04.07_06.43.21-go-javascript-256/2.csv"
python -m deepcode.predict \
  -q ../resources/queries.csv \
  -d "$(lang_path javascript)" "$(lang_path go)" \
  -o "$o" \
  -m "$m"\
  -c "$c"

m="../cache/models/2020.04.07_10.28.37-go-php-256/2.pickle"
c="../cache/models/2020.04.07_10.28.37-go-php-256/config.toml"
o="../cache/models/2020.04.07_10.28.37-go-php-256/2.csv"
python -m deepcode.predict \
  -q ../resources/queries.csv \
  -d "$(lang_path php)" "$(lang_path go)" \
  -o "$o" \
  -m "$m"\
  -c "$c"

m="../cache/models/2020.04.07_17.09.37-go-python-256/2.pickle"
c="../cache/models/2020.04.07_17.09.37-go-python-256/config.toml"
o="../cache/models/2020.04.07_17.09.37-go-python-256/2.csv"
python -m deepcode.predict \
  -q ../resources/queries.csv \
  -d "$(lang_path python)" "$(lang_path go)" \
  -o "$o" \
  -m "$m"\
  -c "$c"

