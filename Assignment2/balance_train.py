import csv,random

reviews = []
for i in range(5):
	reviews.append([])

with open("train.csv","r") as file:
	reader = csv.DictReader(file, delimiter=',')
	for row in reader:
		review = str(row['reviews'])
		rating = int(row['ratings'])
		reviews[rating-1].append(review)

# five_reviews = reviews[4] # reviews with rating 5
# reviews.pop()

for i in range(4):
	factor = int(len(reviews[4])/len(reviews[i]))
	if factor==1:
		factor += 2
	reviews[i] = reviews[i]*factor

# n = int((len(reviews[0])+len(reviews[1])+len(reviews[2])+len(reviews[3]))/4) # 4201
# l = random.sample(five_reviews,n)
# reviews.append(l)

mid_reviews = []
for i in range(5):
	for j in range(len(reviews[i])):
		l = []
		l.append(reviews[i][j])
		l.append(i+1)
		mid_reviews.append(l)

final_reviews = random.sample(mid_reviews,len(mid_reviews))

count = 1
for i in final_reviews:
	i.insert(0, count)
	count += 1 

fields = ['', 'reviews', 'ratings']

with open("train_balanced.csv","w+") as file:
	write = csv.writer(file)
	write.writerow(fields)
	write.writerows(final_reviews) 