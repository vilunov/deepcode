#!/bin/bash

m="../cache/models/2020.04.06_18.24.03-go-java-256/2.pickle"
c="../cache/models/2020.04.06_18.24.03-go-java-256/config.toml"
o="../cache/models/2020.04.06_18.24.03-go-java-256/2.csv"
python -m deepcode.predict \
  -q ../resources/queries.csv \
  -l java -l go \
  -o "$o" \
  -m "$m" \
  -c "$c"

m="../cache/models/2020.04.07_06.43.21-go-javascript-256/2.pickle"
c="../cache/models/2020.04.07_06.43.21-go-javascript-256/config.toml"
o="../cache/models/2020.04.07_06.43.21-go-javascript-256/2.csv"
python -m deepcode.predict \
  -q ../resources/queries.csv \
  -l javascript -l go \
  -o "$o" \
  -m "$m" \
  -c "$c"

m="../cache/models/2020.04.07_10.28.37-go-php-256/2.pickle"
c="../cache/models/2020.04.07_10.28.37-go-php-256/config.toml"
o="../cache/models/2020.04.07_10.28.37-go-php-256/2.csv"
python -m deepcode.predict \
  -q ../resources/queries.csv \
  -l php -l go \
  -o "$o" \
  -m "$m" \
  -c "$c"

m="../cache/models/2020.04.07_17.09.37-go-python-256/2.pickle"
c="../cache/models/2020.04.07_17.09.37-go-python-256/config.toml"
o="../cache/models/2020.04.07_17.09.37-go-python-256/2.csv"
python -m deepcode.predict \
  -q ../resources/queries.csv \
  -l python -l go \
  -o "$o" \
  -m "$m" \
  -c "$c"
