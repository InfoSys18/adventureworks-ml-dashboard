SELECT
	soh.OrderDate
	,soh.OnlineOrderFlag
	,sod.SalesOrderID
	,sod.SalesOrderDetailID
	,sod.ProductID
	,p.Name AS ProductName
	,pc.Name AS Category
	,psc.Name AS Subcategory
	,soh.CustomerID
	,st.Name AS Territory
	,sod.OrderQty
	,sod.UnitPrice
	,sod.UnitPriceDiscount
	,sod.LineTotal
	,p.StandardCost
	FROM Sales.SalesOrderHeader AS soh
	JOIN Sales.SalesOrderDetail AS sod
		ON soh.SalesOrderID = sod.SalesOrderID
	JOIN Production.Product AS p
		ON sod.ProductID = p.ProductID
	JOIN Production.ProductSubcategory AS psc
		ON p.ProductSubcategoryID = psc.ProductSubcategoryID
	JOIN Production.ProductCategory AS pc
		ON psc.ProductCategoryID = pc.ProductCategoryID
	JOIN Sales.SalesTerritory AS st
		ON soh.TerritoryID = st.TerritoryID
		