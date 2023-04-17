a= ["1","32","4"]
b=["4","4","7"]

c = zip(a,b)
ncontent_ntokens = [

         3

        - (1 if len(c) == 0 else 0)
        for h, c in zip(a, b)
    ]

for a in ncontent_ntokens:
    print(a)