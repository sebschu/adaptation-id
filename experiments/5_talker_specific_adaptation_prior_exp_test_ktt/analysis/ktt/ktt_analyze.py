import json, csv, sys

with open(sys.argv[1], "r") as in_f: 
	incsv = csv.reader(in_f)
	next(incsv)

	outcsv = csv.writer(sys.stdout)

	categories_all = {
		'Animals':set(['bear', 'dog', 'horse', 'lion', 'pig', 'wolf']),
		'Colors':set(['blue', 'green', 'orange', 'purple', 'red', 'yellow']),
		'Countries':set(['australia','brazil','china','france','india','japan', 'egypt']),
		'Distances':set(['meter', 'mile', 'inch', 'kilometer', 'foot', 'yard']),
		'Relatives':set(['mother','father','aunt','uncle','brother','sister']),
		"Metals": set(["tin", "nickel", "steel", "lead", "zinc", "iron"])
	}

	outcsv.writerow(['workerid','responseTime','difficulty', 'trialScore','cat1','cat1_correct','cat1_actual','cat2','cat2_correct','cat2_actual','cat3','cat3_correct','cat3_actual','cat4','cat4_correct','cat4_actual','check_manually'])

	for row in incsv:

		workerid = row[4]
		difficulty = row[10]
		answer_to_parse = json.loads(json.loads(row[9]))

		response_time = answer_to_parse['responseTime']
		correct_answers = answer_to_parse['answers']
		participant_answers = answer_to_parse['userResponse']
		numcorrect = answer_to_parse['roundScore']

		#Cat1 Cat2 Cat3 Cat4

		categories = sorted(correct_answers.keys())

		cat1 = categories[0]
		cat1_correct = correct_answers[cat1].lower()
		cat1_actual = participant_answers[cat1].lower()

		cat2 = categories[1]
		cat2_correct = correct_answers[cat2].lower()
		cat2_actual = participant_answers[cat2].lower()


		if len(categories) >= 3:
			cat3 = categories[2]
			cat3_correct = correct_answers[cat3].lower()
			cat3_actual = participant_answers[cat3].lower()
		else:
			cat3 = 'NA'
			cat3_correct = 'NA'
			cat3_actual = 'NA'

		if len(categories) == 4:
			cat4 = categories[3]
			cat4_correct = correct_answers[cat4].lower()
			cat4_actual = participant_answers[cat4].lower()
		else:
			cat4 = 'NA'
			cat4_correct = 'NA'
			cat4_actual = 'NA'

		check_manually = 0

		for category in correct_answers:
			if participant_answers[category].lower() not in categories_all[category]:
				check_manually = 1

		outcsv.writerow([workerid,response_time,difficulty,numcorrect,cat1,cat1_correct,cat1_actual,cat2,cat2_correct,cat2_actual,cat3,cat3_correct,cat3_actual,cat4,cat4_correct,cat4_actual,check_manually])




