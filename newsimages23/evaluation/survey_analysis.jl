using CSV, DataFrames
using CairoMakie


df = CSV.read("qualtrics_results.csv", DataFrame)

#stack and remove empty
stacked = stack(df[:, 1:end-3], :)
stacked = stacked[.!(ismissing.(stacked[:, :value])), :]

stacked[!, :question] = floor.(Int, parse.(Int, stacked[:, :variable]) ./ 10)
stacked[!, :type] = parse.(Int, stacked[:, :variable]) .% 10

ground_truth_ai = ["24", "31", "34", "44", "54", "81", "94", "121", "144", "161", "164"]

fig = Figure()

ax = Axis(fig[1, 1],
xlabel = "Image Source",
ylabel = "Rating",
xticks = (1:4, ["Ground Truth", "Stock", "AI", "CLIP"]),
yticks = -3:3,
xticklabelrotation = pi/4,
title = "")

violin!(stacked[:, :type], stacked[:, :value])

save("violin_all.pdf", fig)


fig = Figure(resolution = (1000, 600))

ax = Axis(fig[1, 1],
#xlabel = "Image Source",
#ylabel = "Rating",
xticks = (1:4, ["Ground Truth", "Stock", "AI", "CLIP"]),
yticks = (-3:3, ["Strongly Disagree", "Disagree", "Somewhat disagree", "Neither agree nor disagree", "Somewhat agree", "Agree", "Strongly agree"]),
title = "")

is_ai = map(x -> x in ground_truth_ai || endswith(x, "3"), stacked[:, :variable])
side = @. ifelse(is_ai == 1, :right, :left)
color = @. ifelse(is_ai == 1, :orange, :teal)

violin!(stacked[:, :type], stacked[:, :value], side = side, color = color)

axislegend(ax, [PolyElement(polycolor = c) for c in [:teal, :orange]], ["Natural", "Generated"],
"Image Type", orientation = :horizontal, framevisible = false, position = :rb)

save("violin_split.pdf", fig)
