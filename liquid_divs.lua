-- liquid_divs.lua
-- A Pandoc Lua filter to convert fenced divs to raw HTML divs.

function Div(div)
  -- Check if the Div has at least one class assigned to it.
  if div.classes[1] then
    -- Get the first class name.
    local class_name = div.classes[1]

    -- Render the content of the Div back into markdown.
    -- This ensures that any markdown inside your environment remains markdown.
    local content = pandoc.write(pandoc.Pandoc(div.content), 'markdown')

    -- Construct the final HTML string.
    local html_block = '<div markdown="1" class="' .. class_name .. '">\n' .. content .. '\n</div>'

    -- Return a RawBlock of HTML, which Pandoc will insert directly
    -- into the markdown output.
    return pandoc.RawBlock('markdown', html_block)
  end
  -- If the div has no classes, let Pandoc handle it normally.
  return nil
end
