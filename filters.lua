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

function Link(link)
  if link.attributes['reference-type'] == 'ref' then
    link.attributes = {} -- This empties the attributes table
  end
  return link
end

-- Put math on its own line
function Math(elem)
  -- Check if the element is a block of display math.
  -- We don't want to modify inline math.
  if elem.mathtype == 'DisplayMath' then
    -- Construct the Kramdown-compatible math block as a string.
    -- It adds a newline after the opening '$$' and before the closing '$$'.
    local kramdown_text = '$$\n' .. elem.text .. '\n$$'

    -- Return a 'RawBlock' of markdown. This tells Pandoc to insert
    -- the raw string directly into the output markdown file, bypassing
    -- the standard markdown writer for this element.
    return pandoc.RawBlock('markdown', kramdown_text)
  end

  -- For any other element type (e.g., InlineMath), return 'nil'
  -- to indicate that no changes should be made.
  return nil
end
