#!/bin/bash
egrep -i 'loss|blue' $1 | grep -v '000 ' | grep -v EOS | sed 'N;N;s/\n/ /g' | cut -f 2,12,15,18 -d' '
