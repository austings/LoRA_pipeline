#!/bin/sh
set -e

VERSION=3.0.1 # setup.py MUST be edited to this version
PACKAGE=tortoise-tts
FILE=$PACKAGE-$VERSION.zip

zip -r $FILE . -x .git/**\* -x *.zip
aws s3 cp $FILE s3://neets/turtles/packages/ --acl public-read

echo "https://neets.s3.amazonaws.com/turtles/packages/$FILE"
