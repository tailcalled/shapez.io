#!/bin/sh
git commit -a
cd gulp
yarn gulp main.build.prod
cd ..
rsync -avz build/* root@personality.surveydata.online:/var/www/personality.surveydata.online/orderliness/shapez.io
#ssh root@personality.surveydata.online command
rsync -avz server/*.py root@personality.surveydata.online:/var/www/personality.surveydata.online/orderliness/