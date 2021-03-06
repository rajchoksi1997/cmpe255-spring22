import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Solution:
    def __init__(self) -> None:
        # TODO: 
        # Load data from data/chipotle.tsv file using Pandas library and 
        # assign the dataset to the 'chipo' variable.
        file = 'data/chipotle.tsv'
        self.chipo = pd.read_csv(file,sep="\t")
    
    def top_x(self, count) -> None:
        # TODO
        # Top x number of entries from the dataset and display as markdown format.
        topx = self.chipo.head(count)
        print(topx.to_markdown())
        
    def count(self) -> int:
        # TODO
        # The number of observations/entries in the dataset.
        return self.chipo.shape[0]
    
    def info(self) -> None:
        # TODO
        # print data info.
        self.chipo.info(verbose=True)
    
    def num_column(self) -> int:
        # TODO return the number of columns in the dataset
        return len(self.chipo.columns)
    
    def print_columns(self) -> None:
        # TODO Print the name of all the columns.
        print("All Columns are...")
        cols = self.chipo.columns
        for col in cols:
            print(col)
    
    def most_ordered_item(self):
        # TODO
        qty=0
        item_name=""
        for name,gp in self.chipo.groupby("item_name"):
            temp = gp.quantity.sum()
            if temp>qty:
                qty=temp
                item_name=name
#         item_name = 
        order_id =self.chipo[self.chipo['item_name']==item_name].order_id.sum()
        quantity = qty
        return item_name, order_id, quantity

    def total_item_orders(self) -> int:
       # TODO How many items were orderd in total?
       return self.chipo.quantity.sum()
   
    def total_sales(self) -> float:
        # TODO 
        # 1. Create a lambda function to change all item prices to float.
        # 2. Calculate total sales.
        self.chipo['temp'] = self.chipo.item_price.apply(lambda x: float(str(x).replace("$","")))
        self.chipo['total_price'] = self.chipo.loc[:,['temp','quantity']].apply(lambda x: x[0]*x[1],axis=1)
        return self.chipo['total_price'].sum()
   
    def num_orders(self) -> int:
        # TODO
        # How many orders were made in the dataset?
        return self.chipo.order_id.nunique()
    
    def average_sales_amount_per_order(self) -> float:
        # TODO
        return round(self.chipo['total_price'].sum()/self.chipo.order_id.nunique(),2)

    def num_different_items_sold(self) -> int:
        # TODO
        # How many different items are sold?
        return self.chipo.item_name.nunique()
    
    def plot_histogram_top_x_popular_items(self, x:int) -> None:
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)
        # TODO
        # 1. convert the dictionary to a DataFrame
        # 2. sort the values from the top to the least value and slice the first 5 items
        # 3. create a 'bar' plot from the DataFrame
        # 4. set the title and labels:
        #     x: Items
        #     y: Number of Orders
        #     title: Most popular items
        # 5. show the plot. Hint: plt.show(block=True).
        d=dict(letter_counter)
        d={k:v for k,v in sorted(d.items(),key=lambda x:-x[1])}
        new_d={}
        for k in list(d.keys())[:x]:
            new_d[k]=d[k]
            
        keys=new_d.keys()
        values=new_d.values()
        plt.figure(figsize=(10, 4), dpi=80)
        plt.xlabel("Items")
        plt.ylabel("Number of Orders")
        plt.bar(keys,values)
    
        
    def scatter_plot_num_items_per_order_price(self) -> None:
        # TODO
        # 1. create a list of prices by removing dollar sign and trailing space.
        # 2. groupby the orders and sum it.
        # 3. create a scatter plot:
        #       x: orders' item price
        #       y: orders' quantity
        #       s: 50
        #       c: blue
        # 4. set the title and labels.
        #       title: Numer of items per order price
        #       x: Order Price
        #       y: Num Items
        self.chipo['item_price'] = self.chipo.item_price.apply(lambda x: float(str(x).strip().replace("$","")))
        self.chipo['item_price'] = self.chipo.loc[:,['item_price','quantity']].apply(lambda x: x[0]*x[1],axis=1)
        tdf = self.chipo.groupby(["order_id"]).sum()
        ax1 = tdf.plot.scatter(x='item_price',y='quantity',c='blue',s=50,figsize=(12, 6))
    
        

def test() -> None:
    solution = Solution()
    solution.top_x(10)
    count = solution.count()
    print(count)
    assert count == 4622
    solution.info()
    count = solution.num_column()
    assert count == 5
    item_name, order_id, quantity = solution.most_ordered_item()
    assert item_name == 'Chicken Bowl'
    assert order_id == 713926	
    assert quantity == 159
    total = solution.total_item_orders()
    assert total == 4972
    assert 39237.02 == solution.total_sales()
    assert 1834 == solution.num_orders()
    assert 21.39 == solution.average_sales_amount_per_order()
    assert 50 == solution.num_different_items_sold()
    solution.plot_histogram_top_x_popular_items(5)
    solution.scatter_plot_num_items_per_order_price()

    
if __name__ == "__main__":
    # execute only if run as a script
    test()
    
    